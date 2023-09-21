import pyrosm
import numpy as np
import time as tm
from graph_tool.all import Graph
from geopy.exc import GeocoderServiceError
from geopy.geocoders import Nominatim

class OSMGraph(Graph):
    def __init__(self, OSM_PATH='.'):
        self.node_coords = {}
        self.graph = self.create_osm_graph(OSM_PATH)

    def download_osm_file(self, OSM_PATH):
        """
        Downloads the latest OSM file for Santiago.

        Parameters:
            OSM_PATH (str): The directory where the OSM file will be saved.

        Returns:
            str: The path to the downloaded OSM file.
        """
        fp = pyrosm.get_data(
            "Santiago",
            update=True,
            directory=OSM_PATH
        )

        return fp

    def create_osm_graph(self, OSM_PATH):
        """
        Creates a graph-tool's graph using the downloaded OSM data for Santiago.

        Returns:
            graph: osm data converted to a graph
        """
        # Download latest OSM data
        fp = self.download_osm_file(OSM_PATH)

        osm = pyrosm.OSM(fp)

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
            self.node_coords[node_id] = (lat, lon)

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
            source_coords = self.node_coords[source_node]
            target_coords = self.node_coords[target_node]
            distance = np.linalg.norm(np.array(source_coords) - np.array(target_coords))

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

    def get_nodes_and_edges(self):
        """
        Returns a tuple containing two lists: one with the nodes and another with the edges.
        """
        nodes = list(self.graph.vertices())
        edges = list(self.graph.edges())
        return nodes, edges

    def print_graph(self):
        """
        Prints the vertices and edges of the graph.
        """
        print("Vertices:")
        for vertex in self.graph.vertices():
            print(f"Vertex ID: {int(vertex)}, lon: {self.graph.vertex_properties['lon'][vertex]}, lat: {self.graph.vertex_properties['lat'][vertex]}")

        print("\nEdges:")
        for edge in self.graph.edges():
            source = int(edge.source())
            target = int(edge.target())
            print(f"Edge: {source} -> {target}")

    def find_node_by_coordinates(self, lon, lat):
        """
        Finds a node in the graph based on its coordinates (lon, lat).

        Parameters:
            lon (float): the longitude of the node.
            lat (float): the latitude of the node.

        Returns:
            vertex: the vertex in the graph with the specified coordinates, or None if not found.
        """
        for vertex in self.graph.vertices():
            if self.graph.vertex_properties["lon"][vertex] == lon and self.graph.vertex_properties["lat"][vertex] == lat:
                return vertex
        return None

    def find_node_by_id(self, node_id):
        """
        Finds a node in the graph based on its id.

        Parameters:
            node_id (long): the id of the node.

        Returns:
            vertex: the vertex in the graph with the specified id, or None if not found.
        """
        for vertex in self.graph.vertices():
            if self.graph.vertex_properties["node_id"][vertex] == node_id:
                return vertex
        return None

    def find_nearest_node(self, latitude, longitude):
        """
        Finds the nearest node in the graph to a given set of coordinates.

        Parameters:
            latitude (float): the latitude of the coordinates.
            longitude (float): the longitude of the coordinates.

        Returns:
            vertex: the vertex in the graph closest to the given coordinates.
        """
        query_point = np.array([longitude, latitude])

        # Obtains vertex properties: 'lon' and 'lat'
        lon_prop = self.graph.vertex_properties['lon']
        lat_prop = self.graph.vertex_properties['lat']

        # Calculates the euclidean distances between the node's coordinates and the consulted address's coordinates
        distances = np.linalg.norm(np.vstack((lon_prop.a, lat_prop.a)).T - query_point, axis=1)

        # Finds the nearest node's index
        nearest_node_index = np.argmin(distances)
        nearest_node = self.graph.vertex(nearest_node_index)

        return nearest_node

    def address_locator(self, address):
        """
        Finds the given address in the OSM graph.

        Parameters:
        address (str): The address to be located.

        Returns:
        int: The ID of the nearest vertex in the graph.

        Raises:
        GeocoderServiceError: If there is an error with the geocoding service.
        """
        geolocator = Nominatim(user_agent="ayatori")
        while True:
            try:
                location = geolocator.geocode(address)
                break
            except GeocoderServiceError:
                i = 0
                if i < 15:
                    print("Geocoding service error. Retrying in 5 seconds...")
                    tm.sleep(5)
                    i+=1
                else:
                    msg = "Error: Too many retries. Geocoding service may be down. Please try again later."
                    print(msg)
                    return
        if location is not None:
            lat, lon = location.latitude, location.longitude
            nearest = self.find_nearest_node(lat, lon)
            return nearest
        msg = "Error: Address couldn't be found."
        print(msg)
