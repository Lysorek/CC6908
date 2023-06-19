import sys
from pathlib import Path

AVES_ROOT = Path("..")

EOD_PATH = AVES_ROOT / "data" / "external" / "EOD_STGO"
OSM_PATH = AVES_ROOT / "data" / "external" / "OSM"


import graph_tool.all as gt
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely.geometry
from pyrosm import get_data, OSM
from pyrosm.data import sources
from datetime import datetime, date, time

from aves.data import eod, census
from aves.features.utils import normalize_rows
from aves.models.network import Network
from aves.visualization.networks import NodeLink
import graph_tool.search
import graph_tool.flow
from aves.visualization.figures import figure_from_geodataframe
from geopy.geocoders import Nominatim
import pygtfs
import os
from graph_tool.all import Graph

# Graph Settings
# Sets the image's quality. The default value is 80.
mpl.rcParams["figure.dpi"] = 192
# Sets the font to be used
mpl.rcParams["font.family"] = "Fira Sans Extra Condensed"


# GTFS
def get_gtfs_data():
    """
    Reads the GTFS data from a file and creates a directed graph with its info, using the 'pygtfs' library. This gives
    the transit feed data of Santiago's public transport, including "Red Metropolitana de Movilidad" (previously known
    as Transantiago), "Metro de Santiago", "EFE Trenes de Chile", and "Buses de Acercamiento Aeropuerto".

    Returns:
        graph: GTFS data converted to a graph.
    """
    # Load GTFS data
    sched = pygtfs.Schedule(":memory:")
    pygtfs.append_feed(sched, "gtfs.zip") # This takes around 2 minutes (01:51.44)

    # Create a graph per route
    graphs = {}
    stop_id_map = {}  # To assign unique ids to every stop

    for route in sched.routes:
        graph = Graph(directed=True)
        stop_ids = set()
        trips = [trip for trip in sched.trips if trip.route_id == route.route_id]

        weight_prop = graph.new_edge_property("int")  # The edge's weight will be the id

        for trip in trips:
            stop_times = trip.stop_times

            for i in range(len(stop_times)):
                stop_id = stop_times[i].stop_id

                if stop_id not in stop_id_map:
                    vertex = graph.add_vertex()  # New empty vertex
                    stop_id_map[stop_id] = vertex  # Assing the new vertex to the stop id
                else:
                    vertex = stop_id_map[stop_id]  # Obtains the existing vertex

                stop_ids.add(vertex)

                if i < len(stop_times) - 1:
                    next_stop_id = stop_times[i + 1].stop_id

                    if next_stop_id not in stop_id_map:
                        next_vertex = graph.add_vertex()  # Adds a new empty vertex for the next stop
                        stop_id_map[next_stop_id] = next_vertex  # Assing the new vertex to the stop id
                    else:
                        next_vertex = stop_id_map[next_stop_id]  # Obtains the existing vertex for the next stop

                    e = graph.add_edge(vertex, next_vertex)  # Adds an edge between the stops
                    weight_prop[e] = 1  # The edge's weight is 1

        graphs[route.route_id] = graph

    # Store graphs into a file
    for route_id, graph in graphs.items():
        weight_prop = graph.new_edge_property("int")  # Creates a new property (edge's weight)

        for e in graph.edges():  # For every edge
            weight_prop[e] = 1  # Assign a weight of 1

        graph.edge_properties["weight"] = weight_prop  # Assing the property to the graph

        graph.save(f"{route_id}.gt")

    #print(stop_id_map.keys())
    return graph


# OSM
def get_osm_data():
    """
    Obtains the required OpenStreetMap data using the 'pyrosm' library. This gives the map info of Santiago.

    Returns:
        graph: osm data converted to a graph.
    """
    fp = get_data(
        "Santiago",
        update=True,
        directory=OSM_PATH # In local testing, "C:/Users/felip/Desktop/Universidad/15° Semestre (Otoño 2023)/CC6909-Trabajo de Título/CC6909-Ayatori"
    ) # This takes around 40 seconds (00:35.06)

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

    print("GETTING NODES")
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

    print("GETTING EDGES")
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

# AUX FUNCTION
def print_graph(graph):
    print("Vertices:")
    for vertex in graph.vertices():
        print(f"Vertex ID: {int(vertex)}, lon: {graph.vertex_properties['lon'][vertex]}, lat: {graph.vertex_properties['lat'][vertex]}")

    print("\nEdges:")
    for edge in graph.edges():
        source = int(edge.source())
        target = int(edge.target())
        print(f"Edge: {source} -> {target}")

graph = get_osm_data()
#print_graph(graph)

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


vertices = graph.vertices()

#EJEMPLO
i = 0
for vertex in vertices:
    # Realiza las operaciones que desees con cada vértice
    # Por ejemplo, puedes acceder a las propiedades del vértice utilizando los diccionarios de propiedades
    if i < 5:
        lon = graph.vertex_properties["lon"][vertex]
        lat = graph.vertex_properties["lat"][vertex]
        print(lon, lat)
        i+=1

lon = -70.636785
lat = -33.4369036

geolocator = Nominatim(user_agent="ayatori")

location = geolocator.geocode("Beauchef 850, Santiago, Chile")

node = find_node_by_coordinates(graph, lon, lat)
if node is not None:
    direccion = geolocator.reverse((lat,lon))
    print("El nodo con coordenadas ({}, {}) fue encontrado en el grafo.".format(lon, lat))
    print("Corresponde a la dirección: {}".format(direccion))
else:
    print("El nodo con coordenadas ({}, {}) no fue encontrado en el grafo.".format(lon, lat))
# FIN EJEMPLO

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

    # Obtener las propiedades de vértice 'lon' y 'lat'
    lon_prop = graph.vertex_properties['lon']
    lat_prop = graph.vertex_properties['lat']

    # Calcular la distancia euclidiana entre las coordenadas del nodo y la consulta
    distances = np.linalg.norm(np.vstack((lon_prop.a, lat_prop.a)).T - query_point, axis=1)

    # Encontrar el índice del nodo más cercano
    nearest_node_index = np.argmin(distances)
    nearest_node = graph.vertex(nearest_node_index)

    return nearest_node

def address_locator(graph, loc):
    location = geolocator.geocode(loc)
    long, lati = location.longitude, location.latitude
    nearest = find_nearest_node(graph,lati,long)
    near_lon, near_lat = graph.vertex_properties["lon"][nearest], graph.vertex_properties["lat"][nearest]
    near_location = geolocator.reverse((near_lat,near_lon))
    near_id = graph.vertex_properties["node_id"][nearest]
    print("Ubicación entregada: {}".format(loc))
    print("Las coordenadas de la ubicación entregada son ({},{})".format(long,lati))
    print("El vértice más cercano a la ubicación entregada está en las coordenadas ({},{})".format(near_lon, near_lat))
    print("Dirección: {}".format(near_location))
    print("El id del nodo es {}".format(near_id))
    return near_id


#address_locator(graph, "Beauchef 850, Santiago")



def connection_scan(graph, source_address, target_address, departure_time, departure_date, max_depth):
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

    connections = []
    current_route = []
    print("HOLA 1")

    def recursive_dfs(vertex, current_time, current_route):
        """
        Performs a recursive Depth-First Search (DFS) from the given vertex with the current time.
        """
        print("HOLA 2")
        departure_seconds = departure_time.hour * 3600 + departure_time.minute * 60 + departure_time.second
        current_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second

        if current_seconds > departure_seconds or len(current_route) > max_depth:
            return
        
        print("HOLA 3")

        if vertex == target_node:
            connections.append(current_route[:])  # Append a copy of current_route to connections
            return
        
        print("HOLA 4")

        out_edges = graph.get_out_edges(vertex)
        
        print("HOLA 5")
        print(out_edges)
        
        for edge in out_edges:
            neighbor = edge.target()
            travel_time = graph.ep['time'][edge]
            arrival_time = current_time + travel_time

            if arrival_time <= departure_time:
                current_route.append(neighbor)
                current_route_copy = current_route[:]
                recursive_dfs(neighbor, arrival_time, current_route_copy)
                current_route.pop()
                
        print("HOLA 6")

    source_node = address_locator(graph, source_address)
    print("HOLA 7")
    target_node = address_locator(graph, target_address)
    print("")
    
    print("HOLA 8")

    recursive_dfs(source_node, departure_time, [source_node])
    
    print("HOLA 9")
    
    return connections

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

    #graph = get_osm_data()

    connection_scan(graph, source_address, target_address, source_hour, source_date, max_depth=1)

# Run
csa_commands()
