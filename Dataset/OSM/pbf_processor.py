import geopandas as gpd
import matplotlib.pyplot as plt # Es necesario importar el plot para que se vea
from pyrosm import get_data, OSM
from pyrosm.data import sources

# COMANDOS #
# cd "Desktop/Universidad/14° Semestre (Primavera 2022)/CC6908-Introducción al Trabajo de Título/CC6909-Ayatori/Dataset/OSM"
# python.exe pbf_processor.py

#print(sources.south_america.available)

fp = get_data(
    "Santiago",
    update=True,
    directory="C:/Users/felip/Desktop/Universidad/14° Semestre (Primavera 2022)/CC6908-Introducción al Trabajo de Título/CC6909-Ayatori"
    )

print("Filepath: ", fp)

osm = OSM(fp)

# Driving
#drive_net = osm.get_network(network_type="driving")
#drive_net.plot()

# Walking
nodes, edges = osm.get_network(nodes=True,network_type="walking")
ax = edges.plot()
nodes.plot(ax = ax, color = 'orange', markersize = 2)

plt.title('Caminos disponibles en Santiago para viajar: caminando.')
plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.show()