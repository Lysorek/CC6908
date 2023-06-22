import sys
import time
from pathlib import Path
from geopy.geocoders import Nominatim
import pygtfs
import os
from graph_tool.all import Graph
from graph_tool.topology import shortest_distance, shortest_path
from pyrosm import get_data, OSM
import graph_tool.all as gt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import shapely.geometry
from pyrosm.data import sources
from datetime import datetime, date, time

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

    column_names_list = list(nodes.columns)
    coordinates = nodes[['lon', 'lat']].values
    ids = nodes['id'].values

    graph = gt.Graph()

    # Create vertex properties for lon and lat
    lon_prop = graph.new_vertex_property("float")
    lat_prop = graph.new_vertex_property("float")
    id_prop = graph.new_vertex_property("long")

    vertex_map = {}

    print("GETTING OSM NODES...")
    for index, row in nodes.iterrows():
        lon = row['lon']
        lat = row['lat']
        node_id = row['id']

        vertex = graph.add_vertex()
        vertex_map[node_id] = vertex

        # Assigning node properties
        lon_prop[vertex] = lon
        lat_prop[vertex] = lat
        id_prop[vertex] = node_id

    # Assign the properties to the graph
    graph.vertex_properties["lon"] = lon_prop
    graph.vertex_properties["lat"] = lat_prop
    graph.vertex_properties["node_id"] = id_prop

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

        graph.add_edge(source_vertex, target_vertex)

    print("OSM DATA HAS BEEN SUCCESSFULLY RECEIVED")
    return graph

# OSM Graph
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
    #print("Ubicación entregada: {}".format(loc))
    print("Las coordenadas de la ubicación entregada son ({},{})".format(long,lati))
    print("El vértice más cercano a la ubicación entregada está en las coordenadas ({},{})".format(near_lon, near_lat))
    print("Dirección: {}".format(near_location))
    print("El id del nodo es {}".format(near_id))
    return nearest


def create_node_id_mapping(graph):
    node_id_mapping = {}
    node_id_prop = graph.vertex_properties["node_id"]
    for v in graph.vertices():
        node_id = node_id_prop[v]
        node_id_mapping[node_id] = int(v)
    return node_id_mapping

end = time.time()
exec_time = (end-start) / 60
print("ALL THE INFO IS READY. EXECUTION TIME: {} MINUTES".format(exec_time))

### CODE: CSA ALGORITHM ###

# Image's quality. Default: 80
mpl.rcParams["figure.dpi"] = 192
# Fonts
mpl.rcParams["font.family"] = "Fira Sans Extra Condensed"

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

    source_node_id = address_locator(graph, source_address)
    target_node_id = address_locator(graph, target_address)

    source_node = node_id_mapping[source_node_id]
    target_node = node_id_mapping[target_node_id]
    print("ADDRESSES FOUND")
    print("SOURCE NODE: {}. TARGET NODE: {}.".format(source_node, target_node))
    print("DEPARTURE TIME: {}".format(departure_time))

    path = shortest_path(graph, source_node, target_node)

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

    result = connection_scan(osm_graph, source_address, target_address, source_hour, source_date)
    return result


csa_commands()
