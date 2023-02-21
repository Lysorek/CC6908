import os
import geopandas as gpd
from pyrosm import get_data, OSM

#fp = get_data("Santiago")
#print(fp)

filepath = get_data("test_pbf",directory="C:/Users/felip/Desktop/Universidad/14° Semestre (Primavera 2022)/CC6908-Introducción al Trabajo de Título/CC6909-Ayatori")
osm = OSM(filepath)

walk_net = osm.get_network(network_type="walking")
walk_net.plot()