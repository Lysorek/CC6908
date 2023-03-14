import geopandas as gpd
import matplotlib.pyplot as plt # Es necesario importar el plot para que se vea
from pyrosm import get_data, OSM
from pyrosm.data import sources
import networkx as nx
import osmnx as ox


# COMANDOS #
# cd "Desktop/Universidad/14° Semestre (Primavera 2022)/CC6908-Introducción al Trabajo de Título/CC6909-Ayatori/Dataset/OSM"
# cd Dataset/OSM
# python.exe pbf_processor.py

#print(sources.south_america.available)

fp = get_data(
    "Santiago",
    update=True,
    directory="C:/Users/felip/Desktop/Universidad/14° Semestre (Primavera 2022)/CC6908-Introducción al Trabajo de Título/CC6909-Ayatori"
    )

print("Filepath: ", fp)

osm = OSM(fp)

# RUTA MÁS CORTA #
#nodes, edges = osm.get_network(nodes=True)
#G = osm.to_graph(nodes, edges, graph_type="networkx")

#source_address = "Beauchef 850, Santiago" # Campus Beauchef de la Universidad de Chile
#target_address = "Av. Sta. Rosa 11315, La Pintana" # Campus Sur de la Universidad de Chile

#source = ox.geocode(source_address)
#target = ox.geocode(target_address)

#source_node = ox.nearest_nodes(G, source[1], source[0])
#target_node = ox.nearest_nodes(G, target[1], target[0])

#route = nx.shortest_path(G, source_node, target_node, weight="length")
#fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')


# GRÁFICOS GENERALES #

# Driving
#nodes, edges = osm.get_network(nodes=True, network_type="driving")
#plt.title('Caminos disponibles en Santiago para viajar: en auto.')

# Cycling
#nodes, edges = osm.get_network(nodes=True, network_type="cycling")
#plt.title('Caminos disponibles en Santiago para viajar: en bicicleta.')

# Walking
nodes, edges = osm.get_network(nodes=True,network_type="walking")

ax = edges.plot()
#nodes.plot(ax = ax, color = 'orange', markersize = 2)

plt.title('Caminos disponibles en Santiago para viajar: caminando.')
plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.show()
