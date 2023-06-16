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

class Visitor(graph_tool.search.BFSVisitor):
    def __init__(self, node_id, edge_weight, pred, dist):
        self.pred = pred
        self.dist = dist
        self.cost = edge_weight

        self.root = node_id
        self.dist[node_id] = 0

        self.next_ring = dict()
        self.visited = set()
        self.visited.add(node_id)

    def discover_vertex(self, u):    
        #print("-->", u, "has been discovered!")
        self.next_ring[u] = self.dist[u]
        #print(self.next_ring)

    def examine_vertex(self, u):
        #print(u, "has been examined...")
        #print(self.next_ring)
        pass

    def tree_edge(self, e):
        self.pred[e.target()] = int(e.source())
        
        cost = self.dist[e.source()] + self.cost[e]

        # TODO: quizás hay que seleccionar un costo porque hay varias maneras de llegar
        if not e.target() in self.visited:
            self.dist[e.target()] = cost
            self.visited.add(e.target())
            
        #print(f"{e.source()} --> {e.target()}: {self.cost[e]}", " - tree edge")

    def finish_vertex(self, u):
        del self.next_ring[u]
        print("-->", u, "has been finished!", self.next_ring)
        

        if all(cost > 1500 for cost in self.next_ring.values()):
            for node in self.next_ring.keys():
                self.dist[node] = -1
            raise graph_tool.search.StopSearch()



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

    vertex_map = {}
    for index, row in nodes.iterrows():
        print(row)
        lon = row['lon']
        lat = row['lat']

        vertex = graph.add_vertex()
        vertex_map[index] = vertex


        #print("Guardando la información de "+str(index))
        # Asignar las coordenadas a las propiedades del vértice
        lon_prop[vertex] = lon
        lat_prop[vertex] = lat

    # Assign the lon and lat properties to the graph
    graph.vertex_properties["lon"] = lon_prop
    graph.vertex_properties["lat"] = lat_prop


    for index, row in edges.iterrows():
        #print(row)
        source_node = row['u']
        target_node = row['v']

        if row["length"] < 2 or source_node == "" or target_node == "":
            continue # Skip edges with empty or missing nodes

        if source_node not in vertex_map or target_node not in vertex_map:
            #print(f"Skipping edge with missing nodes: {source_node} -> {target_node}")
            continue  # Skip edges with missing nodes

        source_vertex = vertex_map[source_node]
        target_vertex = vertex_map[target_node]

        if not graph.vertex(source_vertex) or not graph.vertex(target_vertex):
            print(f"Skipping edge with non-existent vertices: {source_vertex} -> {target_vertex}")
            continue  # Skip edges with non-existent vertices

        graph.add_edge(source_vertex, target_vertex)

    return graph


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
print_graph(graph)

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

# EJEMPLO
#lon = -33.4577725
#lat = -70.6635288

#node = find_node_by_coordinates(graph, lon, lat)
#if node is not None:
#    print("El nodo con coordenadas ({}, {}) fue encontrado en el grafo.".format(lon, lat))
#else:
#    print("El nodo con coordenadas ({}, {}) no fue encontrado en el grafo.".format(lon, lat))


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

    Returns:
        list: the list of travel connections needed to arrive destination.
    """

    def get_node_from_address(address):
        """
        Obtains the node in the graph corresponding to the given address.

        Args:
            address (string): the address to find the corresponding node for.

        Returns:
            node: the node in the graph corresponding to the address.
        """
        geolocator = Nominatim(user_agent="ayatori")

        location = geolocator.geocode(address)

        # Use the address as the node identifier
        node = location.address

        return node
    
    connections = []
    visited = set()

    def recursive_dfs(vertex, current_time, current_route):
        """
        Performs a recursive Depth-First Search (DFS) from the given vertex with the current time.
        """
        if current_time > departure_time:
            return

        visited.add(vertex)

        if vertex == target_node:
            connections.append(current_route.copy())
            return

        out_neighbors = graph.get_out_neighbors(vertex)
        for neighbor in out_neighbors:
            if neighbor not in visited:
                travel_time = graph.ep['time'][vertex, neighbor]
                arrival_time = current_time + travel_time

                current_route.append(neighbor)
                recursive_dfs(neighbor, arrival_time, current_route)
                current_route.pop()

        visited.remove(vertex)

    source_node = get_node_from_address(source_address)
    target_node = get_node_from_address(target_address)

    recursive_dfs(source_node, departure_time, [source_node])
    return connections



def csa_commands():
    """
    Process the inputs given by the user to run the Connection Scan Algorithm.
    """

    # System's date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Fecha y hora actuales =", dt_string)

    # Date formatting
    today = date.today()
    today_format = today.strftime("%d/%m/%Y")

    # Time formatting
    moment = now.strftime("%H:%M:%S")

    # User inputs
    # Date and time
    source_date = input(
        "Ingresa la fecha del viaje en formato DD/MM/YYY (presiona Enter para usar la fecha actual) : ") or today_format
    print(source_date)
    source_hour = input(
        "Ingresa la hora del viaje en formato HH:MM:SS (presiona Enter para usar la hora actual) : ") or moment
    print(source_hour)

    # Source address
    source_example = "Beauchef 850, Santiago"
    while True:
        source_address = input(
            "Ingresa dirección de inicio (Ejemplo: 'Beauchef 850, Santiago'. Presiona Enter para usarlo): ") or source_example
        if source_address.strip() != '':
            print("Dirección de Inicio ingresada: " + source_address)
            break

    # Destination address
    destination_example = "Pio Nono 1, Providencia"
    while True:
        target_address = input(
            "Ingresa dirección de destino (Ejemplo: 'Pio Nono 1, Providencia'. Presiona Enter para usarlo): ") or destination_example
        if target_address.strip() != '':
            print("Dirección de Destino ingresada: " + target_address)
            break

    print("Preparando ruta, por favor espere...")

    graph = get_osm_data()

    connection_scan(graph, source_address, target_address, source_hour, source_date)


# Run
csa_commands()
