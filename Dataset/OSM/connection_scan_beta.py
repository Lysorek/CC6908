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

# Graph Settings
# Sets the image's quality. The default value is 80.
mpl.rcParams["figure.dpi"] = 192
# Sets the font to be used
mpl.rcParams["font.family"] = "Fira Sans Extra Condensed"


def get_osm_data():
    """
    Obtains the required OpenStreetMap data using the 'pyrosm' library. This gives the map info of Santiago.

    Returns:
        graph: osm data converted to a graph
    """
    fp = get_data(
        "Santiago",
        update=True,
        directory=OSM_PATH # In local testing, "C:/Users/felip/Desktop/Universidad/15° Semestre (Otoño 2023)/CC6909-Trabajo de Título/CC6909-Ayatori"
    )

    print("Filepath: ", fp)
    osm = OSM(fp)

    nodes, edges = osm.get_network(nodes=True)

    graph = gt.Graph()

    vertex_map = {}
    for node in nodes:
        vertex_map[node] = graph.add_vertex()

    for edge in edges:
        if len(edge) < 2 or edge[0] == "" or edge[1] == "":
            continue  # Skip edges with empty or missing nodes
        source_node = edge[0]
        target_node = edge[1]
        if source_node not in vertex_map or target_node not in vertex_map:
            continue  # Skip edges with missing nodes
        source_vertex = vertex_map[source_node]
        target_vertex = vertex_map[target_node]
        graph.add_edge(source_vertex, target_vertex)

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

    def get_vertex_from_address(address):
        """
        Obtains the vertex in the graph corresponding to the given address.

        Args:
            address (string): the address to find the corresponding vertex for.

        Returns:
            vertex: the vertex in the graph corresponding to the address.
        """
        # I need to implement here the logic to access the OSM data

    connections = []
    visited = set()

    def recursive_dfs(vertex, current_time, current_route):
        """
        Performs a recursive Depth-First Search (DFS) from the given vertex with the current time.
        """
        if current_time > departure_time:
            return

        visited.add(vertex)

        if vertex == target_vertex:
            connections.append(current_route.copy())
            return

        for neighbor in vertex.out_neighbors():
            if neighbor not in visited:
                travel_time = graph.ep['time'][graph.edge(vertex, neighbor)]
                arrival_time = current_time + travel_time

                current_route.append(neighbor)
                recursive_dfs(neighbor, arrival_time, current_route)
                current_route.pop()

        visited.remove(vertex)

    source_vertex = get_vertex_from_address(source_address)
    target_vertex = get_vertex_from_address(target_address)

    recursive_dfs(source_vertex, departure_time, [source_vertex])
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


# Ejemplo de uso
# Crear un grafo de ejemplo
""" graph = gt.Graph()
vA = graph.add_vertex()
vB = graph.add_vertex()
vC = graph.add_vertex()
vD = graph.add_vertex()
vE = graph.add_vertex()

eAB = graph.add_edge(vA, vB)
eBC = graph.add_edge(vB, vC)
eBD = graph.add_edge(vB, vD)
eCD = graph.add_edge(vC, vD)
eDE = graph.add_edge(vD, vE)

graph.ep['time'] = graph.new_edge_property("int")
graph.ep['time'][eAB] = 5
graph.ep['time'][eBC] = 4
graph.ep['time'][eBD] = 3
graph.ep['time'][eCD] = 2
graph.ep['time'][eDE] = 4
 """
# Realizar una consulta
""" source = vA
destination = vE
departure_time = 0
result = connection_scan(graph, source, destination, departure_time)

# Imprimir las rutas encontradas
print(
    f"Rutas desde el nodo {source} al nodo {destination} desde el tiempo {departure_time}:")
for idx, route in enumerate(result):
    print(f"Ruta {idx+1}: {' -> '.join(map(str, route))}")
 """

csa_commands()