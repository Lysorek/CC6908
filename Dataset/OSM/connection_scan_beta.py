import graph_tool.all as gt
from datetime import datetime, date, time

def connection_scan(graph, source, destination, departure_time, date, hour):
    """
    Aplica el algoritmo Connection Scan para buscar rutas de viaje desde la fuente (source) hasta el destino (destination)
    dado un tiempo de partida (departure_time). Se solicita también la fecha del viaje (date) y la hora de inicio (hour).
    Por defecto, se utilizan la fecha y hora actuales del sistema. (Por implementar).
    """
    connections = []
    visited = set()

    def recursive_dfs(node, current_time):
        """
        Realiza una búsqueda en profundidad (DFS) recursiva desde el nodo dado (node) con un tiempo actual (current_time).
        """
        if current_time > departure_time:
            return

        visited.add(node)

        if node == destination:
            connections.append([])
            return

        for neighbor in node.out_neighbors():
            if neighbor not in visited:
                travel_time = graph.ep['time'][graph.edge(node, neighbor)]
                arrival_time = current_time + travel_time
                recursive_dfs(neighbor, arrival_time)

        visited.remove(node)

    recursive_dfs(source, departure_time)
    return connections

# Ejemplo de uso
# Crear un grafo de ejemplo
graph = gt.Graph()
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

# Realizar una consulta
source = vA
destination = vE
departure_time = 0
result = connection_scan(graph, source, destination, departure_time)

# Imprimir las rutas encontradas
print(f"Rutas desde el nodo {source} al nodo {destination} desde el tiempo {departure_time}:")
for idx, route in enumerate(result):
    print(f"Ruta {idx+1}: {' -> '.join(map(str, route))}")