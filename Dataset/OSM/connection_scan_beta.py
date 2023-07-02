import sys
import time
from pathlib import Path
from geopy.geocoders import Nominatim
import pygtfs
import os
from graph_tool.all import *
from graph_tool.topology import shortest_distance, shortest_path
from pyrosm import get_data, OSM
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import shapely.geometry
from datetime import datetime, date
from pyrosm.data import sources
from queue import Queue
import heapq
from collections import defaultdict
import graphviz

from aves.data import eod, census
from aves.features.utils import normalize_rows

### CODE: OBTAINING DATA AND AUX FUNCTIONS ###

start = time.time()
print("GETTING INFO")

# PATHS
AVES_ROOT = Path("..")
EOD_PATH = AVES_ROOT / "data" / "external" / "EOD_STGO"
OSM_PATH = AVES_ROOT / "data" / "external" / "OSM"

## GTFS ##

def get_gtfs_data():
    """
    Reads the GTFS data from a file and creates a directed graph with its info, using the 'pygtfs' library. This gives
    the transit feed data of Santiago's public transport, including "Red Metropolitana de Movilidad" (previously known
    as Transantiago), "Metro de Santiago", "EFE Trenes de Chile", and "Buses de Acercamiento Aeropuerto".

    Returns:
        graph: GTFS data converted to a graph.
    """
    # Create a new schedule object using a GTFS file
    sched = pygtfs.Schedule(":memory:")
    pygtfs.append_feed(sched, "gtfs.zip") # This takes around 2 minutes (01:51.44)

    # Create a graph per route
    graphs = {}
    stop_id_map = {}  # To assign unique ids to every stop

    print("GETTING GTFS ROUTES...")
    for route in sched.routes:
        graph = Graph(directed=True)
        stop_ids = set()
        trips = [trip for trip in sched.trips if trip.route_id == route.route_id]

        weight_prop = graph.new_edge_property("int")  # Propiedad para almacenar los pesos de las aristas

        for trip in trips:
            stop_times = trip.stop_times

            for i in range(len(stop_times)):
                stop_id = stop_times[i].stop_id

                if stop_id not in stop_id_map:
                    vertex = graph.add_vertex()  # Añadir un vértice vacío
                    stop_id_map[stop_id] = vertex  # Asignar el vértice al identificador de parada
                else:
                    vertex = stop_id_map[stop_id]  # Obtener el vértice existente

                stop_ids.add(vertex)

                if i < len(stop_times) - 1:
                    next_stop_id = stop_times[i + 1].stop_id

                    if next_stop_id not in stop_id_map:
                        next_vertex = graph.add_vertex()  # Añadir un vértice vacío para la siguiente parada
                        stop_id_map[next_stop_id] = next_vertex  # Asignar el vértice al identificador de parada
                    else:
                        next_vertex = stop_id_map[next_stop_id]  # Obtener el vértice existente para la siguiente parada

                    e = graph.add_edge(vertex, next_vertex)  # Añadir una arista entre las paradas
                    weight_prop[e] = 1  # Asignar peso 1 a la arista

        graphs[route.route_id] = graph

    print("DONE")
    print("STORING ROUTE GRAPHS...")

    # Store graphs into a file
    for route_id, graph in graphs.items():
        weight_prop = graph.new_edge_property("int")  # Crear una nueva propiedad de peso de arista

        for e in graph.edges():  # Iterar sobre las aristas del grafo
            weight_prop[e] = 1  # Asignar el peso 1 a cada arista

        graph.edge_properties["weight"] = weight_prop  # Asignar la propiedad de peso al grafo

        graph.save(f"{route_id}.gt")

    print("GTFS DATA RECEIVED SUCCESSFULLY")
    return graph

# GTFS Graph
gtfs_graph = get_gtfs_data()


## OSM ##

def get_osm_data():
    """
    Obtains the required OpenStreetMap data using the 'pyrosm' library. This gives the map info of Santiago.

    Returns:
        graph: osm data converted to a graph
    """
    # Download latest OSM data
    fp = get_data(
        "Santiago",
        update=True,
        directory=OSM_PATH
    )

    osm = OSM(fp)

    nodes, edges = osm.get_network(nodes=True)

    graph = Graph()

    # Create vertex properties for lon and lat
    lon_prop = graph.new_vertex_property("float")
    lat_prop = graph.new_vertex_property("float")

    # Create properties for the ids
    # Every OSM node has its unique id, different from the one given in the graph
    node_id_prop = graph.new_vertex_property("long")
    graph_id_prop = graph.new_vertex_property("long")

    # Create edge properties
    u_prop = graph.new_edge_property("long")
    v_prop = graph.new_edge_property("long")
    length_prop = graph.new_edge_property("double")
    weight_prop = graph.new_edge_property("double")

    vertex_map = {}

    print("GETTING OSM NODES...")
    for index, row in nodes.iterrows():
        lon = row['lon']
        lat = row['lat']
        node_id = row['id']
        graph_id = index
        node_coords[node_id] = (lat, lon)

        vertex = graph.add_vertex()
        vertex_map[node_id] = vertex

        # Assigning node properties
        lon_prop[vertex] = lon
        lat_prop[vertex] = lat
        node_id_prop[vertex] = node_id
        graph_id_prop[vertex] = graph_id

    # Assign the properties to the graph
    graph.vertex_properties["lon"] = lon_prop
    graph.vertex_properties["lat"] = lat_prop
    graph.vertex_properties["node_id"] = node_id_prop
    graph.vertex_properties["graph_id"] = graph_id_prop

    print("DONE")
    print("GETTING OSM EDGES...")

    for index, row in edges.iterrows():
        source_node = row['u']
        target_node = row['v']

        if row["length"] < 2 or source_node == "" or target_node == "":
            continue # Skip edges with empty or missing nodes

        if source_node not in vertex_map or target_node not in vertex_map:
            print(f"Skipping edge with missing nodes: {source_node} -> {target_node}")
            continue  # Skip edges with missing nodes

        source_vertex = vertex_map[source_node]
        target_vertex = vertex_map[target_node]

        if not graph.vertex(source_vertex) or not graph.vertex(target_vertex):
            print(f"Skipping edge with non-existent vertices: {source_vertex} -> {target_vertex}")
            continue  # Skip edges with non-existent vertices

        # Calculate the distance between the nodes and use it as the weight of the edge
        source_coords = node_coords[source_node]
        target_coords = node_coords[target_node]
        distance = abs(source_coords[0] - target_coords[0]) + abs(source_coords[1] - target_coords[1])

        e = graph.add_edge(source_vertex, target_vertex)
        u_prop[e] = source_node
        v_prop[e] = target_node
        length_prop[e] = row["length"]
        weight_prop[e] = distance

    graph.edge_properties["u"] = u_prop
    graph.edge_properties["v"] = v_prop
    graph.edge_properties["length"] = length_prop
    graph.edge_properties["weight"] = weight_prop

    print("OSM DATA HAS BEEN SUCCESSFULLY RECEIVED")
    return graph


# OSM Graph
node_coords = {}
osm_graph = get_osm_data()
osm_vertices = osm_graph.vertices()

# AUX FUNCTION FOR DEBUGGING
def print_graph(graph):
    print("Vertices:")
    for vertex in graph.vertices():
        print(f"Vertex ID: {int(vertex)}, lon: {graph.vertex_properties['lon'][vertex]}, lat: {graph.vertex_properties['lat'][vertex]}")

    print("\nEdges:")
    for edge in graph.edges():
        source = int(edge.source())
        target = int(edge.target())
        print(f"Edge: {source} -> {target}")

# AUX FUNCTIONS TO FIND NODES
def find_node_by_coordinates(graph, lon, lat):
    """
    Finds a node in the graph based on its coordinates (lon, lat).

    Args:
        graph (graph): the graph containing the node coordinates.
        lon (float): the longitude of the node.
        lat (float): the latitude of the node.

    Returns:
        vertex: the vertex in the graph with the specified coordinates, or None if not found.
    """
    for vertex in graph.vertices():
        if graph.vertex_properties["lon"][vertex] == lon and graph.vertex_properties["lat"][vertex] == lat:
            return vertex
    return None

def find_node_by_id(graph, node_id):
    """
    Finds a node in the graph based on its id.

    Args:
        graph (graph): the graph containing the node coordinates.
        node_id (long): the id of the node.

    Returns:
        vertex: the vertex in the graph with the specified id, or None if not found.
    """
    for vertex in graph.vertices():
        if graph.vertex_properties["node_id"][vertex] == node_id:
            return vertex
    return None

def find_nearest_node(graph, latitude, longitude):
    query_point = np.array([longitude, latitude])

    # Obtains vertex properties: 'lon' and 'lat'
    lon_prop = graph.vertex_properties['lon']
    lat_prop = graph.vertex_properties['lat']

    # Calculates the euclidean distances between the node's coordinates and the consulted address's coordinates
    distances = np.linalg.norm(np.vstack((lon_prop.a, lat_prop.a)).T - query_point, axis=1)

    # Finds the nearest node's index
    nearest_node_index = np.argmin(distances)
    nearest_node = graph.vertex(nearest_node_index)

    return nearest_node

# Finds the given address in the OSM graph
def address_locator(graph, loc):
    geolocator = Nominatim(user_agent="ayatori")
    location = geolocator.geocode(loc)
    long, lati = location.longitude, location.latitude
    nearest = find_nearest_node(graph,lati,long)
    near_lon, near_lat = graph.vertex_properties["lon"][nearest], graph.vertex_properties["lat"][nearest]
    near_location = geolocator.reverse((near_lat,near_lon))
    near_id = graph.vertex_properties["node_id"][nearest]
    graph_id = graph.vertex_properties["graph_id"][nearest]
    #print("Ubicación entregada: {}".format(loc))
    print("Las coordenadas de la ubicación entregada son ({},{})".format(long,lati))
    print("El vértice más cercano a la ubicación entregada está en las coordenadas ({},{})".format(near_lon, near_lat))
    print("Dirección: {}".format(near_location))
    print("El id del nodo es {}".format(near_id))
    print("El id en el grafo es {}".format(graph_id))
    return nearest

def get_largest_component(component_sizes):
    largest_size = max(component_sizes)
    largest_component = [i for i, size in enumerate(component_sizes) if size == largest_size]
    return largest_component


def analyze_connectivity(graph):
    # Verificar si el grafo es conexo
    is_graph_connected = graph.num_edges() == graph.num_vertices() - 1
    print("El grafo es conexo:", is_graph_connected)

    # Obtener los componentes conectados utilizando la función label_components()
    components = label_components(graph)[0]

    # Contar el número de componentes conectados
    num_components = len(set(components))
    print("Número de componentes conectados:", num_components)
    #print("Componentes conectados:", set(components))

    # Obtener el tamaño de cada componente
    component_sizes = []
    for component_id in set(components):
        size = np.sum(components == component_id)
        component_sizes.append(size)

    #print("Tamaño de cada componente:")
    #for component_id, size in enumerate(component_sizes):
    #    print("Componente {}: {}".format(component_id, size))

    # Obtener los componentes aislados
    isolated_components = [i for i, size in enumerate(component_sizes) if size == 1]
    print("Número de componentes aislados:", len(isolated_components))
    #print("Componentes aislados:", isolated_components)

    # Obtener el componente más grande
    largest_component = get_largest_component(component_sizes)
    largest_component_size = largest_component[0]
    print("Componente más grande: tamaño:", largest_component_size)


# Analizar la conectividad del grafo
analyze_connectivity(osm_graph)

def create_node_id_mapping(graph):
    node_id_mapping = {}
    node_id_prop = graph.vertex_properties["node_id"]
    for v in graph.vertices():
        node_id = node_id_prop[v]
        node_id_mapping[int(v)] = node_id
    return node_id_mapping

def create_edge_mapping(graph):
    edge_mapping = {}
    node_id_prop = graph.vertex_properties["graph_id"]
    for e in graph.edges():
        source_vertex, target_vertex = e.source(), e.target()
        source_node_id = node_id_prop[source_vertex]
        target_node_id = node_id_prop[target_vertex]
        edge_index = graph.edge_index[e]
        edge_mapping[edge_index] = (source_node_id, target_node_id)
    return edge_mapping

#node_mapping = create_node_id_mapping(osm_graph)

#edge_mapping = create_edge_mapping(osm_graph)#678892 709089
#for edge_index, (source_node, target_node) in edge_mapping.items():
#    print("Edge {}: {} -> {}".format(edge_index, source_node, target_node))

def edge_count(graph, vertex):
    v = graph.vertex(vertex)
    out_degree = v.out_degree()
    in_degree = v.in_degree()
    degree = out_degree + in_degree
    return degree

#v = 709089  # Índice del vértice deseado
#degree = edge_count(osm_graph, v)
#print("Número de aristas entrantes al vértice {}: {}".format(v, v.in_degree()))
#print("Número de aristas salientes del vértice {}: {}".format(v, v.out_degree()))
#print("Número de aristas total del nodo {}: {}".format(v, degree))

def make_undirected(graph):
    undirected_graph = Graph(directed=False)
    vprop_map = graph.new_vertex_property("object")

    # Create vertex properties for lon and lat
    lon_prop = undirected_graph.new_vertex_property("float")
    lat_prop = undirected_graph.new_vertex_property("float")
    node_id_prop = undirected_graph.new_vertex_property("long")
    graph_id_prop = undirected_graph.new_vertex_property("long")

    # Create edge properties
    u_prop = undirected_graph.new_edge_property("long")
    v_prop = undirected_graph.new_edge_property("long")
    length_prop = undirected_graph.new_edge_property("double")
    weight_prop = undirected_graph.new_edge_property("double")

    undirected_vertex_map = {}

    for v in graph.vertices():
        new_v = undirected_graph.add_vertex()
        vprop_map[new_v] = v
        lon = graph.vertex_properties["lon"][v]
        lat = graph.vertex_properties["lat"][v]
        node_id = graph.vertex_properties["node_id"][v]
        graph_id = graph.vertex_properties["graph_id"][v]

        undirected_vertex_map[node_id] = new_v
        #print("NODO {} EN GRAFO {}".format(node_id, graph_id))

        # Assigning node properties
        lon_prop[new_v] = lon
        lat_prop[new_v] = lat
        node_id_prop[new_v] = node_id
        graph_id_prop[new_v] = graph_id

    # Assign the properties to the graph
    undirected_graph.vertex_properties["lon"] = lon_prop
    undirected_graph.vertex_properties["lat"] = lat_prop
    undirected_graph.vertex_properties["node_id"] = node_id_prop
    undirected_graph.vertex_properties["graph_id"] = graph_id_prop


    for e in graph.edges():
        source, target = e.source(), e.target()
        source_node = graph.edge_properties["u"][e]
        target_node = graph.edge_properties["v"][e]
        lgt = graph.edge_properties["length"][e]
        wt = graph.edge_properties["weight"][e]

        if lgt < 2 or source_node == "" or target_node == "":
            continue # Skip edges with empty or missing nodes

        if source_node not in undirected_vertex_map or target_node not in undirected_vertex_map:
            print(f"Skipping edge with missing nodes: {source_node} -> {target_node}")
            continue  # Skip edges with missing nodes

        source_vertex = undirected_vertex_map[source_node]
        target_vertex = undirected_vertex_map[target_node]

        if not undirected_graph.vertex(source_vertex) or not undirected_graph.vertex(target_vertex):
            print(f"Skipping edge with non-existent vertices: {source_vertex} -> {target_vertex}")
            continue  # Skip edges with non-existent vertices

        e = undirected_graph.add_edge(source_vertex, target_vertex)
        u_prop[e] = source_node
        v_prop[e] = target_node
        length_prop[e] = lgt
        weight_prop[e] = wt

    undirected_graph.edge_properties["u"] = u_prop
    undirected_graph.edge_properties["v"] = v_prop
    undirected_graph.edge_properties["length"] = length_prop
    undirected_graph.edge_properties["weight"] = weight_prop

    return undirected_graph

# Convertir el grafo en no dirigido
undirected_graph = make_undirected(osm_graph)


end = time.time()
exec_time = (end-start) / 60
print("ALL THE INFO IS READY. EXECUTION TIME: {} MINUTES".format(exec_time))

### CODE: CSA ALGORITHM ###

# Image's quality. Default: 80
mpl.rcParams["figure.dpi"] = 192
# Fonts
mpl.rcParams["font.family"] = "Fira Sans Extra Condensed"

def dijkstra(graph, start, end):
    """
    Find the shortest path between two nodes in a graph using Dijkstra's algorithm.

    Args:
        graph (dict): the graph represented as a dictionary of nodes and their neighbors.
        start (int): the starting node.
        end (int): the ending node.

    Returns:
        list: the shortest path between the starting and ending nodes.
    """
    # Initialize the distances and visited nodes
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    visited = set()

    # Initialize the priority queue with the starting node
    pq = [(0, start, [])]

    while pq:
        # Get the node with the smallest distance from the starting node
        (dist, node, path) = heapq.heappop(pq)

        # If we've already visited this node, skip it
        if node in visited:
            continue

        # Add the node to the visited set
        visited.add(node)

        # Add the node to the path
        path = path + [node]

        # If we've reached the end node, return the path
        if node == end:
            return path

        # Update the distances of the neighbors of the current node
        for neighbor, weight in graph[node].items():
            if neighbor not in visited:
                new_distance = dist + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(pq, (new_distance, neighbor, path))

    # If we didn't find a path, return an empty list
    return []


def connection_scan(graph, source_address, target_address, departure_time, departure_date):
    """
    The Connection Scan algorithm is applied to search for travel routes from the source to the destination,
    given a departure time and date. By default, the algorithm uses the current date and time of the system.
    However, you can specify a different date if needed (to be implemented).

    Args:
        graph (graph): the graph used to visualize the travel routes.
        source_address (string): the source address of the travel.
        target_address (string): the destination address of the travel.
        departure_time (time): the time at which the travel should start.
        departure_date (date): the date on which the travel should be done.
        max_depth (int): the maximum depth of the search.

    Returns:
        list: the list of travel connections needed to arrive at the destination.
    """
    node_id_mapping = create_node_id_mapping(graph)

    source_node = address_locator(graph, source_address)
    target_node = address_locator(graph, target_address)

    # Convert source and target node IDs to integers
    source_node_graph_id = graph.vertex_properties["graph_id"][source_node]
    target_node_graph_id = graph.vertex_properties["graph_id"][target_node]

    print("ADDRESSES FOUND")
    print("SOURCE NODE: {}. TARGET NODE: {}.".format(source_node_graph_id, target_node_graph_id))
    print("DEPARTURE TIME: {}".format(departure_time))

    # Convert the graph to a dictionary of nodes and their neighbors
    #graph_dict = defaultdict(dict)
    #for edge in graph.edges():
    #    u, v = edge.source(), edge.target()
    #    weight = graph.edge_properties["weight"][edge]
    #    graph_dict[u][v] = weight

    # Find the shortest path using Dijkstra's algorithm
    #path = dijkstra(graph_dict, source_node_graph_id, target_node_graph_id)

    # Convert the path to a list of node IDs
    #path = [graph.vertex(node_id) for node_id in path]
    #path = [graph.vertex_properties["graph_id"][node] for node in path]

    #print("RECONSTRUCTED PATH:")
    #for v in path:
    #    print(node_id_mapping[v], end=" -> ")
    #print()

    path = shortest_path(graph, source_node_graph_id, target_node_graph_id)

    return path


def csa_commands():
    """
    Process the inputs given by the user to run the Connection Scan Algorithm.
    """

    # System's date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    #print("Fecha y hora actuales =", dt_string)

    # Date formatting
    today = date.today()
    today_format = today.strftime("%d/%m/%Y")

    # Time formatting
    moment = now.strftime("%H:%M:%S")
    used_time = datetime.strptime(moment, "%H:%M:%S").time()

    # User inputs
    # Date and time
    source_date = input(
        "Ingresa la fecha del viaje en formato DD/MM/YYY (presiona Enter para usar la fecha actual) : ") or today_format
    print(source_date)
    source_hour = input(
        "Ingresa la hora del viaje en formato HH:MM:SS (presiona Enter para usar la hora actual) : ") or used_time
    print(source_hour)
    ## MODIFICAR ESTO

    # Source address
    source_example = "Beauchef 850, Santiago"
    while True:
        source_address = input(
            "Ingresa dirección de inicio (Ejemplo: 'Beauchef 850, Santiago'. Presiona Enter para usarlo): ") or source_example
        if source_address.strip() != '':
            #print("Dirección de Inicio ingresada: " + source_address)
            break

    # Destination address
    destination_example = "Pio Nono 1, Providencia"
    while True:
        target_address = input(
            "Ingresa dirección de destino (Ejemplo: 'Pio Nono 1, Providencia'. Presiona Enter para usarlo): ") or destination_example
        if target_address.strip() != '':
            #print("Dirección de Destino ingresada: " + target_address)
            break

    print("Preparando ruta, por favor espere...")

    #print("CON GRAFO ORIGINAL")
    #result1 = connection_scan(osm_graph, source_address, target_address, source_hour, source_date)
    print("CON GRAFO MODIFICADO")
    path = connection_scan(undirected_graph, source_address, target_address, source_hour, source_date)
    path_nodes = path[0]
    path_edges = path[1]
    geolocator = Nominatim(user_agent="ayatori")
    nod = []
    path_coords = []
    for node in path_nodes:
        lon, lat = undirected_graph.vertex_properties["lon"][node], undirected_graph.vertex_properties["lat"][node]
        location = geolocator.reverse((lat,lon))
        path_coords.append((lat, lon))
        if location not in nod:
            nod.append(location)
            print(location)

    # Convert the graph to a Graphviz graph
    gv_graph = Graph(engine='neato', graph_attr={'size': '8,8', 'overlap': 'false'}, node_attr={'shape': 'point', 'width': '0.05', 'color': 'black'}, edge_attr={'penwidth': '0.5', 'color': 'gray'})
    for v in undirected_graph.vertices():
        gv_graph.node(str(int(v)), pos='{},{}'.format(undirected_graph.vertex_properties['lon'][v], undirected_graph.vertex_properties['lat'][v]))
    for e in undirected_graph.edges():
        gv_graph.edge(str(int(e.source())), str(int(e.target())), weight=str(undirected_graph.edge_properties['weight'][e]))

    # Convert the node indices in the `path` list to match the node indices in the `gv_graph` graph
    path = [str(int(node)) for node in path_nodes]

    # Highlight the shortest path in the graph
    for i in range(len(path) - 1):
        gv_graph.edge(path[i], path[i+1], color='red', penwidth='2.0')

    # Render the graph to a file or display it in a Jupyter Notebook
    #gv_graph.render('graph.png', view=True)

    return path

csa_commands()
