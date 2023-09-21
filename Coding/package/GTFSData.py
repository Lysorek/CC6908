import pygtfs
import os
import pandas as pd
from math import *
from datetime import datetime, date, time, timedelta
from graph_tool.all import Graph

class GTFSData:
    def __init__(self, GTFS_PATH='gtfs.zip'):
        self.scheduler = self.create_scheduler(GTFS_PATH)
        self.graphs = {}
        self.route_stops = {}
        self.special_dates = []
        self.graphs, self.route_stops, self.special_dates = self.get_gtfs_data()

    def create_scheduler(self, GTFS_PATH):
        """
        Creates the scheduler for the class, using the GTFS file, located in the given path directory.

        Parameters:
        GTFS_PATH (PATH): the path where the GTFS file is located.

        Returns:
        pygtfs.Schedule: the scheduler object
        """
        scheduler = pygtfs.Schedule(":memory:")
        pygtfs.append_feed(scheduler, GTFS_PATH)
        return scheduler

    def get_gtfs_data(self):
        """
        Reads the GTFS data from a file and creates a directed graph with its info, using the 'pygtfs' library. This gives
        the transit feed data of Santiago's public transport, including "Red Metropolitana de Movilidad" (previously known
        as Transantiago), "Metro de Santiago", "EFE Trenes de Chile", and "Buses de Acercamiento Aeropuerto".

        Returns:
            graphs: GTFS data converted to a dictionary of graphs, one per route.
            route_stops: Dictionary containing the stops for each route.
            special_dates: List of special calendar dates.
        """
        sched = self.scheduler

        # Get special calendar dates
        for cal_date in sched.service_exceptions: # Calendar_dates is renamed in pygtfs
            self.special_dates.append(cal_date.date.strftime("%d/%m/%Y"))

        stop_id_map = {} # To assign unique ids to every stop
        stop_coords = {}

        for route in sched.routes:
            graph = Graph(directed=True)
            stop_ids = set()
            trips = [trip for trip in sched.trips if trip.route_id == route.route_id]

            # Create a new vertex property for node_id
            node_id_prop = graph.new_vertex_property("string")

            # Create edge properties
            u_prop = graph.new_edge_property("object")
            v_prop = graph.new_edge_property("object")
            weight_prop = graph.new_edge_property("int")
            graph.edge_properties["weight"] = weight_prop
            graph.edge_properties["u"] = u_prop
            graph.edge_properties["v"] = v_prop

            added_edges = set() # To keep track of the edges that have already been added

            for trip in trips:
                stop_times = trip.stop_times
                orientation = trip.trip_id.split("-")[1]

                for i in range(len(stop_times)):
                    stop_id = stop_times[i].stop_id
                    sequence = stop_times[i].stop_sequence

                    if stop_id not in stop_id_map:
                        vertex = graph.add_vertex()
                        stop_id_map[stop_id] = vertex
                    else:
                        vertex = stop_id_map[stop_id]

                    stop_ids.add(vertex)

                    # Assign the node_id property to the vertex
                    node_id_prop[vertex] = stop_id

                    if i < len(stop_times) - 1:
                        next_stop_id = stop_times[i + 1].stop_id

                        if next_stop_id not in stop_id_map:
                            next_vertex = graph.add_vertex()
                            stop_id_map[next_stop_id] = next_vertex
                        else:
                            next_vertex = stop_id_map[next_stop_id]

                        edge = (vertex, next_vertex)
                        if edge not in added_edges: # Check if the edge has already been added
                            e = graph.add_edge(*edge)
                            graph.edge_properties["weight"][e] = 1
                            graph.edge_properties["u"][e] = node_id_prop[vertex]
                            graph.edge_properties["v"][e] = node_id_prop[next_vertex]
                            added_edges.add(edge) # Add the edge to the set of added edges

                        if route.route_id not in stop_coords:
                            stop_coords[route.route_id] = {}

                        if stop_id not in stop_coords[route.route_id]:
                            stop = sched.stops_by_id(stop_id)[0]
                            stop_coords[route.route_id][stop_id] = (stop.stop_lon, stop.stop_lat)

                            if route.route_id not in self.route_stops:
                                self.route_stops[route.route_id] = {}

                            self.route_stops[route.route_id][stop_id] = {
                                "route_id": route.route_id,
                                "stop_id": stop_id,
                                "coordinates": stop_coords[route.route_id][stop_id],
                                "orientation": "round" if orientation == "I" else "return",
                                "sequence": sequence,
                                "arrival_times": []
                            }

                    arrival_time = (datetime.min + stop_times[i].arrival_time).time()

                    if stop_id in self.route_stops[route.route_id]:
                        self.route_stops[route.route_id][stop_id]["arrival_times"].append(arrival_time)

            # Assign the node_id property to the graph
            graph.vertex_properties["node_id"] = node_id_prop

            self.graphs[route.route_id] = graph

            stops_by_direction = {"round_trip": [], "return_trip": []}
            for trip in trips:
                stop_times = trip.stop_times
                stops = [stop_times[i].stop_id for i in range(len(stop_times))]

                if trip.direction_id == 0:
                    stops_by_direction["round_trip"].extend(stops)
                else:
                    stops_by_direction["return_trip"].extend(stops)

            round_trip_stops = set(stops_by_direction["round_trip"])
            return_trip_stops = set(stops_by_direction["return_trip"])

            for stop_id in round_trip_stops:
                if stop_id in stop_coords[route.route_id]:
                    if stop_id in self.route_stops[route.route_id]:
                        self.route_stops[route.route_id][stop_id]["orientation"] = "round"
                    else:
                        self.route_stops[route.route_id][stop_id] = {
                            "route_id": route.route_id,
                            "stop_id": stop_id,
                            "coordinates": stop_coords[route.route_id][stop_id],
                            "orientation": "round",
                            "sequence": sequence,
                            "arrival_times": []
                        }

            for stop_id in return_trip_stops:
                if stop_id in stop_coords[route.route_id]:
                    if stop_id in self.route_stops[route.route_id]:
                        self.route_stops[route.route_id][stop_id]["orientation"] = "return"
                    else:
                        self.route_stops[route.route_id][stop_id] = {
                            "route_id": route.route_id,
                            "stop_id": stop_id,
                            "coordinates": stop_coords[route.route_id][stop_id],
                            "orientation": "return",
                            "sequence": sequence,
                            "arrival_times": []
                        }

            # Add edges between every pair of consecutive stops in stop_id_map
            #stop_list = list(stop_id_map.values())
            #for i in range(len(stop_list) - 1):
            #    for j in range(i + 1, len(stop_list)):
            #        edge = (stop_list[i], stop_list[j])
            #        if edge not in added_edges: # Check if the edge has already been added
            #            e = graph.add_edge(*edge)
            #            graph.edge_properties["weight"][e] = 1
            #            graph.edge_properties["u"][e] = node_id_prop[stop_list[i]]
            #            graph.edge_properties["v"][e] = node_id_prop[stop_list[j]]
            #            added_edges.add(edge) # Add the edge to the set of added edges

        for route_id, graph in self.graphs.items():
            weight_prop = graph.new_edge_property("int")

            for e in graph.edges():
                weight_prop[e] = 1

            graph.edge_properties["weight"] = weight_prop

            data_dir = "gtfs_routes"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            graph.save(f"{data_dir}/{route_id}.gt")

        print("GTFS DATA RECEIVED SUCCESSFULLY")

        return self.graphs, self.route_stops, self.special_dates

    def get_route_graph(self, route_id):
        """
        Given a route_id, returns the vertices and edges for the corresponding graph.

        Parameters:
        route_id (str): The ID of the route.

        Returns:
        tuple: A tuple containing the vertices and edges of the graph. The vertices are a list of node IDs, and the edges are a list of tuples containing the source and target node IDs.
        """
        if route_id not in self.graphs:
            print(f"Route {route_id} does not exist.")
            return None

        graph = self.graphs[route_id]
        vertices = []
        for v in graph.vertices():
            node_id = graph.vertex_properties["node_id"][v]
            if node_id != '' and node_id is not None:
                vertices.append(node_id)

        edges = []
        for e in graph.edges():
            u = graph.edge_properties["u"][e]
            v = graph.edge_properties["v"][e]
            if u is not None and v is not None:
                edges.append((u, v))

        return vertices, edges

    def get_route_graph_vertices(self, route_id):
        """
        Given a route_id, returns the vertices for the corresponding graph.

        Parameters:
        route_id (str): The ID of the route.

        Returns:
        list: A list containing the vertices of the graph. The vertices are a list of node IDs.
        """
        if route_id not in self.graphs:
            print(f"Route {route_id} does not exist.")
            return None

        graph = self.graphs[route_id]
        vertices = [graph.vertex_properties["node_id"][v] for v in graph.vertices()]

        return vertices

    def get_route_graph_edges(self, route_id):
        """
        Given a route_id, returns the edges for the corresponding graph.

        Parameters:
        route_id (str): The ID of the route.

        Returns:
        list: A list containing the edges of the graph.
        """
        if route_id not in self.graphs:
            print(f"Route {route_id} does not exist.")
            return None

        graph = self.graphs[route_id]
        edges = [(graph.edge_properties["u"][e], graph.edge_properties["v"][e]) for e in graph.edges()]

        return edges

    def get_route_coordinates(self, route_id):
        round_trip_stops = []
        return_trip_stops = []
        for stop_info in self.route_stops[route_id].values():
            if stop_info["orientation"] == "round":
                round_trip_stops.append(stop_info)
            elif stop_info["orientation"] == "return":
                return_trip_stops.append(stop_info)

        round_trip_stops.sort(key=lambda stop: stop["sequence"])
        return_trip_stops.sort(key=lambda stop: stop["sequence"])

        round_trip_coords = [(stop_info["coordinates"][1], stop_info["coordinates"][0]) for stop_info in round_trip_stops]
        return_trip_coords = [(stop_info["coordinates"][1], stop_info["coordinates"][0]) for stop_info in return_trip_stops]

        return round_trip_coords, return_trip_coords

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points on the earth (specified in decimal degrees).

        Parameters:
        lon1 (float): Longitude of the first point in decimal degrees.
        lat1 (float): Latitude of the first point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.

        Returns:
        float: The distance between the two points in kilometers.
        """
        R = 6372.8  # Earth radius in kilometers
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat / 2)**2 + cos(lat1) * cos(lat2) * sin(dLon / 2)**2
        c = 2 * asin(sqrt(a))
        return R * c

    def get_stop_coords(self, stop_id):
        """
        Given a stop ID, returns the coordinates of the stop with the given ID.
        If the stop ID is not found, returns None.

        Parameters:
        stop_id (int): The ID of the stop to get the coordinates for.

        Returns:
        tuple: A tuple of two floats representing the longitude and latitude of the stop with the given ID.
        None: If the stop ID is not found.
        """
        for route_id, stops in self.route_stops.items():
            for stop_info in stops.values():
                if stop_info["stop_id"] == stop_id:
                    return stop_info["coordinates"]
        return None

    def get_near_stop_ids(self, coords, margin):
        """
        Given a tuple of coordinates and a margin, returns a list of stop IDs
        that are within the specified margin of the given coordinates, along with their orientations.

        Parameters:
        coords (tuple): A tuple of two floats representing the longitude and latitude of the coordinates to search around.
        margin (float): The maximum distance (in kilometers) from the given coordinates to include stops in the result.

        Returns:
        tuple: A tuple of two lists. The first list contains the stop IDs that are within the specified margin of the given coordinates.
        The second list contains tuples of stop IDs and their orientations.
        """
        stop_ids = []
        orientations = []
        for route_id, stops in self.route_stops.items():
            for stop_info in stops.values():
                stop_coords = stop_info["coordinates"]
                distance = self.haversine(coords[1], coords[0], stop_coords[1], stop_coords[0])
                if distance <= margin:
                    orientation = stop_info["orientation"]
                    stop_id = stop_info["stop_id"]
                    if stop_id not in stop_ids:
                        stop_ids.append(stop_id)
                        orientations.append((stop_id, orientation))
        return stop_ids, orientations

    def get_route_stop_ids(self, route_id):
        """
        Given a route ID, returns a list of stop IDs for the stops on the given route.

        Parameters:
        route_id (int): The ID of the route to get the stops for.

        Returns:
        list: A list of stop IDs for the stops on the given route.
        """
        stops = self.route_stops.get(route_id, {})
        return stops.keys()

    def route_stop_matcher(self, route_id, stop_id):
        """
        Given a route ID, and a stop ID, returns True if the stop ID is on the given route,
        and False otherwise.

        Parameters:
        route_id (int): The ID of the route to check.
        stop_id (int): The ID of the stop to check.

        Returns:
        bool: True if the stop ID is on the given route, False otherwise.
        """
        stop_list = self.get_route_stop_ids(route_id)
        return (stop_id in stop_list)

    def is_route_near_coordinates(self, route_id, coordinates, margin):
        """
        Given a route ID, a tuple of coordinates, and a margin, returns True if the route
        has a stop within the specified margin of the given coordinates, and False otherwise.

        Parameters:
        route_id (int): The ID of the route to check.
        coordinates (tuple): A tuple of two floats representing the longitude and latitude of the coordinates to search around.
        margin (float): The maximum distance (in kilometers) from the given coordinates to include stops in the result.

        Returns:
        bool: True if the route has a stop within the specified margin of the given coordinates, False otherwise.
        """
        for stop_info in self.route_stops[route_id].values():
            stop_coords = stop_info["coordinates"]
            distance = self.haversine(coordinates[1], coordinates[0], stop_coords[1], stop_coords[0])
            if distance <= margin:
                return route_id
        return False

    def get_bus_orientation(self, route_id, stop_id):
        """
        Checks and confirms the bus orientation, while visiting a stop, in the GTFS data files.

        Parameters:
        route_id (str): The route or service's ID to check.
        stop_id (str): The visited stop ID.

        Returns:
        str or list: The bus orientation(s) associated with the route_id and stop_id. None if nothing is found.
        """
        stop_times = pd.read_csv("stop_times.txt")
        filtered_stop_times = stop_times[(stop_times["trip_id"].str.startswith(route_id)) & (stop_times["stop_id"] == stop_id)]

        orientations = []
        for trip_id in filtered_stop_times["trip_id"]:
            orientation = trip_id.split("-")[1]
            if orientation == "I" and "round" not in orientations:
                orientations.append("round")
            elif orientation == "R" and "return" not in orientations:
                orientations.append("return")

        if len(orientations) == 0:
            return None
        elif len(set(orientations)) == 1:
            return orientations[0]
        else:
            return orientations

    def connection_finder(self, stop_id_1, stop_id_2):
        """
        Finds all routes that have stops at both given stop IDs.

        Parameters:
        stop_id_1 (str): The ID of the first stop to check.
        stop_id_2 (str): The ID of the second stop to check.

        Returns:
        list: A list of route IDs that have stops at both given stop IDs.
        """
        connected_routes = []
        for route_id, stops in self.route_stops.items():
            stop_ids = [stop_info["stop_id"] for stop_info in stops.values()]

            if stop_id_1 in stop_ids and stop_id_2 in stop_ids:
                connected_routes.append(route_id)
        return connected_routes

    def get_routes_at_stop(self, stop_id):
        """
        Finds all routes that have a stop at the given stop ID.

        Parameters:
        stop_id (str): The ID of the stop to check.

        Returns:
        list: A list of route IDs that have a stop at the given stop ID.
        """
        routes = [route_id for route_id in self.route_stops.keys() if stop_id in self.get_route_stop_ids(route_id) and self.connection_finder(stop_id, stop_id)]
        return routes

    def is_24_hour_service(self, route_id):
        """
        Determines if the given route has a 24-hour service.

        Parameters:
        route_id (str): A string representing the ID of the route.

        Returns:
        bool: True if the route has a 24-hour service, False otherwise.
        """
        # Read the frequencies for the route
        frequencies = pd.read_csv("frequencies.txt")
        route_str = str(route_id) + "-"
        route_frequencies = frequencies[frequencies["trip_id"].str.startswith(route_str)]

        # Check if any frequency has a start time of "00:00:00" and an end time of "24:00:00"
        has_start_time = False
        has_end_time = False
        for _, row in route_frequencies.iterrows():
            start_time = row["start_time"]
            end_time = row["end_time"]
            if start_time == "00:00:00":
                has_start_time = True
            if end_time == "24:00:00":
                has_end_time = True

        return has_start_time and has_end_time

    def check_night_routes(self, valid_services, is_nighttime):
        """
        Filters the given list of route IDs to only include night routes if is_nighttime is True.

        Parameters:
        valid_services (list): A list of route IDs to filter.
        is_nighttime (bool): True if it is nighttime, False otherwise.

        Returns:
        list: A list of route IDs that are night routes if is_nighttime is True, or all route IDs otherwise.
        """
        if is_nighttime:
            #nighttime_routes = [route_id for route_id in valid_services if route_id.endswith("N")]
            nighttime_routes = [route_id for route_id in valid_services if route_id.endswith("N") or self.is_24_hour_service(route_id)]
            if nighttime_routes:
                return nighttime_routes
            else:
                return None
        else:
            daytime_routes = [route_id for route_id in valid_services if not route_id.endswith("N")]
            if daytime_routes:
                return daytime_routes
            else:
                return None

    def is_nighttime(self, source_hour):
        """
        Determines if the given hour is during the nighttime.

        Parameters:
        source_hour (datetime.time): The hour to check.

        Returns:
        bool: True if the hour is during the nighttime, False otherwise.
        """
        start_time = time(0, 0, 0)
        end_time = time(5, 30, 0)
        if start_time <= source_hour <= end_time:
            return True
        else:
            return False

    def is_holiday(self, date_string):
        """
        Checks if a given date is a holiday.

        Parameters:
        date_string (str): A string representing the date in the format "dd/mm/yyyy".

        Returns:
        bool: True if the date is a holiday, False otherwise.
        """
        # Local holidays
        if date_string in self.special_dates:
            return True
        date_obj = datetime.strptime(date_string, "%d/%m/%Y")

        # Weekend days
        day_of_week = date_obj.weekday()
        if day_of_week == 5 or day_of_week == 6:
            return True
        return False

    def is_rush_hour(self, source_hour):
        """
        Determines if the given hour is during rush hour.

        Parameters:
        source_hour (datetime.time): The hour to check.

        Returns:
        bool: True if the hour is during rush hour, False otherwise.
        """
        am_start_time = time(5, 30, 0)
        am_end_time = time(9, 0, 0)
        pm_start_time = time(17, 30, 0)
        pm_end_time = time(21, 0, 0)
        if am_start_time <= source_hour <= am_end_time or pm_start_time <= source_hour <= pm_end_time:
            return True
        else:
            return False

    def check_express_routes(self, valid_services, is_rush_hour):
        """
        Filters the given list of route IDs to only include express routes if is_rush_hour is True.

        Parameters:
        valid_services (list): A list of route IDs to filter.
        is_rush_hour (bool): True if it is rush hour, False otherwise.

        Returns:
        list: A list of route IDs that are express routes if is_rush_hour is True, or all route IDs otherwise.
        """
        if is_rush_hour:
            return valid_services
        else:
            regular_hour_routes = [route_id for route_id in valid_services if not route_id.endswith("e")]
            return regular_hour_routes

    def get_trip_day_suffix(self, date):
        """
        Based on the given date, gets the corresponding trip day suffix for the trip IDs.

        Parameters:
        date (date): The date to be checked.

        Returns
        str: A string with the trip day suffix.
        """
        date_object = datetime.strptime(date, "%d/%m/%Y")
        day_of_week = date_object.weekday()

        if day_of_week < 5:
            trip_day_suffix = "L"
        elif day_of_week == 5:
            trip_day_suffix = "S"
        else:
            trip_day_suffix = "D"

        return trip_day_suffix

    def get_arrival_times(self, route_id, stop_id, source_date):
        """
        Returns the arrival times for a given route and stop.

        Parameters:
        route_id (str): A string representing the ID of the route.
        stop_id (str): A string representing the ID of the stop.

        Returns:
        tuple: A tuple containing a string representing the bus orientation ("round" or "return") and a list of datetime objects representing the arrival times.
        """
        # Read the frequencies.txt file
        frequencies = pd.read_csv("frequencies.txt")

        # Filter the frequencies for the given route ID
        route_frequencies = frequencies[frequencies["trip_id"].str.startswith(route_id)]

        # Get the day suffix
        day_suffix = self.get_trip_day_suffix(source_date)

        # Get the arrival times for the stop for each trip
        stop_route_times = []
        bus_orientation = ""
        for _, row in route_frequencies.iterrows():
            start_time = pd.Timestamp(row["start_time"])
            if row["end_time"] == "24:00:00":
                end_time = pd.Timestamp("23:59:59")
            else:
                end_time = pd.Timestamp(row["end_time"])
            headway_secs = row["headway_secs"]
            round_trip_id = f"{route_id}-I-{day_suffix}"
            return_trip_id = f"{route_id}-R-{day_suffix}"
            round_stop_times = pd.read_csv("stop_times.txt").query(f"trip_id.str.startswith('{round_trip_id}') and stop_id == '{stop_id}'")
            return_stop_times = pd.read_csv("stop_times.txt").query(f"trip_id.str.startswith('{return_trip_id}') and stop_id == '{stop_id}'")
            if len(round_stop_times) == 0 and len(return_stop_times) == 0:
                return
            elif len(round_stop_times) > 0:
                bus_orientation = "round"
                stop_time = pd.Timestamp(round_stop_times.iloc[0]["arrival_time"])
            elif len(return_stop_times) > 0:
                bus_orientation = "return"
                stop_time = pd.Timestamp(return_stop_times.iloc[0]["arrival_time"])
            for freq_time in pd.date_range(start_time, end_time, freq=f"{headway_secs}s"):
                freq_time_str = freq_time.strftime("%H:%M:%S")
                freq_time = datetime.strptime(freq_time_str, "%H:%M:%S")
                stop_route_time = datetime.combine(datetime.min, stop_time.time()) + timedelta(seconds=(freq_time - datetime.min).seconds)
                if stop_route_time not in stop_route_times:
                    stop_route_times.append(stop_route_time)
                stop_time += pd.Timedelta(seconds=headway_secs)

        return bus_orientation, stop_route_times


    def get_time_until_next_bus(self, arrival_times, source_hour, source_date):
        """
        Returns the time until the next three buses.

        Parameters:
        arrival_times (list): A list of datetime objects representing the arrival times of the buses.
        source_hour (datetime.time): The source hour to compare with the arrival times.
        source_date (datetime.date): The source date to check if there are buses remaining.

        Returns:
        list: A list of tuples representing the time until the next three buses in minutes and seconds.
        """
        arrival_times_remaining = []
        for a_time in arrival_times:
            if a_time.time() >= source_hour:
                arrival_times_remaining.append(a_time)
        #arrival_times_remaining = [time for time in arrival_times if time.time() >= source_hour]
        if len(arrival_times_remaining) == 0:
            return None
        else:
            # Sort the remaining arrival times in ascending order
            arrival_times_remaining.sort()

            # Get the datetime objects for the next three buses
            next_buses = []
            for i in range(min(3, len(arrival_times_remaining))):
                next_arrival_time = arrival_times_remaining[i]
                next_bus = datetime.combine(next_arrival_time.date(), next_arrival_time.time())
                next_buses.append(next_bus)

            if next_buses is None:
                print("No buses remaining for the specified date.")
            else:
                # Calculate the time until the next three buses
                time_until_next_buses = []
                for next_bus in next_buses:
                    time_until_next_bus = (next_bus - datetime.combine(next_bus.date(), source_hour)).total_seconds()
                    minutes, seconds = divmod(time_until_next_bus, 60)
                    time_until_next_buses.append((int(minutes), int(seconds)))

                return time_until_next_buses

    def timedelta_to_hhmm(self, td):
        """
        Converts a timedelta object to a string in HHMM format.

        Parameters:
        td (timedelta): The timedelta object to be converted.

        Returns:
        str: A formated string with the time.
        """
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours:02d}:{minutes:02d}"

    def timedelta_separator(self, td):
        """
        Separates a timedelta object into minutes and seconds.

        Parameters:
        td (timedelta): A timedelta object representing a duration of time.

        Returns:
        tuple: A tuple containing the number of minutes and seconds in the timedelta object. The minutes and seconds are both integers.
        """
        total_seconds = td.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return minutes, seconds

    def get_travel_time(self, trip_id, stop_ids):
        """
        Returns the travel time between two stops for a given trip.

        Parameters:
        trip_id (str): A string representing the ID of the trip.
        stop_ids (list): A list of two strings representing the IDs of the stops.

        Returns:
        timedelta: A timedelta object representing the travel time.
        """
        stop_times = pd.read_csv("stop_times.txt").query(f"trip_id.str.startswith('{trip_id}') and stop_id in {stop_ids}")
        if len(stop_times) < 2:
            return None
        arrival_times = [datetime.strptime(arrival_time, "%H:%M:%S") for arrival_time in stop_times["arrival_time"]]
        travel_time = arrival_times[1] - arrival_times[0]
        return travel_time

    def get_trip_sequence(self, route_id, stop_id):
        """
        Given a dictionary of routes and stops, a route ID and a stop ID, gets the trip sequence number corresponding to the stop.

        Parameters:
        route_id (str): The route or service's ID.
        stop_id (str): The stop's ID.

        Returns:
        str: A string representing the sequence number.
        """
        seq = self.route_stops[route_id][stop_id]["sequence"]
        return seq

    def walking_travel_time(self, stop_coords, location_coords, speed):
        """
        Calculates the walking travel time between a location and a stop, given a speed value.

        Parameters:
        stop_coords (tuple): A tuple with the stop's coordinates.
        location_coords (tuple):  A tuple with the location's coordinates.
        speed (float): The walking speed value.

        Returns.
        float: The time (in seconds) that represents the travel time.
        """
        distance = self.haversine(stop_coords[0], stop_coords[1], location_coords[0], location_coords[1])
        time = round((distance / speed) * 3600,2)
        return time

    def parse_metro_stations(self, stops_file):
        """
        Parses the Metro Stations data, creating a dictionary with their names.

        Parameters:
        stops_file (File): The GTFS file with the stop data (stops.txt).

        Returns:
        dict: A dictionary with the names of the stations.
        """
        subway_stops = {}
        with open(stops_file, 'r') as f:
            for line in f:
                stop_id, _, stop_name, _, _, _, _ = line.strip().split(',')
                if stop_id.isdigit():
                    subway_stops[stop_id] = stop_name
        return subway_stops

    def is_metro_station(self, stop_id, route_dict):
        """
        Checks if a stop is a Metro station.

        Parameters:
        stop_id (str): The stop's ID to be checked.
        route_dict (dict): The dictionary with the Metro stations names.

        Returns:
        str or None: A string with the stop ID if the stop is a Metro station, or None if it isn't.
        """
        try:
            route_num = int(stop_id)
            return route_dict[stop_id]
        except ValueError:
            return None
