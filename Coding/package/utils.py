def find_nearest_nodes(osm_graph, gtfs_data, route_id, desired_orientation):
    # Checks the desired orientation validity
    if desired_orientation != "round" and desired_orientation != "return":
        # Invalid orientation
        return

    # Get the stops for the specified route
    stops = gtfs_data.route_stops.get(route_id, {})

    # Filter the stops that are visited on the desired orientation
    trip_stops = [stop_info for stop_info in stops.values() if stop_info["orientation"] == desired_orientation]

    # Find the nearest node for each stop
    nearest_nodes = []
    for stop_info in trip_stops:
        stop_coords = stop_info["coordinates"]
        nearest_node = osm_graph.find_nearest_node(stop_coords[1], stop_coords[0])
        nearest_nodes.append(nearest_node)

    return nearest_nodes
