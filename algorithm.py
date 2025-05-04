import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import itertools

# Load the road network data
def load_road_network(file_path):
    """Load the road network from a GPKG file."""
    nodes = gpd.read_file(file_path, layer='nodes')
    edges = gpd.read_file(file_path, layer='edges')
    star_locations = gpd.read_file(file_path, layer='star_locations')
    
    print(f"Loaded {len(nodes)} nodes, {len(edges)} edges, and {len(star_locations)} star locations")
    return nodes, edges, star_locations

# Create a graph from the road network with realistic travel times
def create_graph(nodes, edges):
    """Create a networkx graph from nodes and edges."""
    G = nx.Graph()
    
    # Add nodes
    for idx, node in nodes.iterrows():
        G.add_node(node['osmid'], geometry=node['geometry'])
    
    # Add edges with properly scaled lengths
    for idx, edge in edges.iterrows():
        u, v = edge['u'], edge['v']
        
        # Scale length appropriately to get realistic travel times
        # 8 m/s = 480 m/min, so a 1 minute drive covers 480 meters
        # The original lengths are in km, so multiply by 1000 to get meters
        length_meters = float(edge['length']) * 1000
        
        # Calculate travel time in minutes
        travel_time_minutes = length_meters / (8 * 60)  # 8 m/s = 480 m/min
        
        # Cap travel time to be realistic for this small area
        travel_time_minutes = min(travel_time_minutes, 5)
        
        # Use name if available
        name = str(edge['name']) if pd.notna(edge['name']) else f"Road {u}-{v}"
        
        # Add edge to graph
        G.add_edge(u, v, length=length_meters, time=travel_time_minutes, name=name)
    
    print(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

# Find the nodes closest to star locations
def find_star_nodes(G, star_locations, nodes):
    """Find graph nodes closest to each star location."""
    star_nodes = []
    
    for idx, star in star_locations.iterrows():
        min_dist = float('inf')
        closest_node = None
        star_geom = star['geometry']
        
        for _, node in nodes.iterrows():
            node_id = node['osmid']
            if node_id in G:  # Ensure node exists in our graph
                dist = star_geom.distance(node['geometry'])
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node_id
        
        if closest_node:
            star_nodes.append(closest_node)
            print(f"Star location {idx+1} mapped to node {closest_node}")
    
    return star_nodes

# Split roads into morning and afternoon with balanced coverage
def split_roads_balanced(G, star_nodes):
    """Split roads between morning and afternoon to ensure balanced coverage."""
    # Split star nodes for morning and afternoon
    morning_stars = star_nodes[:2]
    afternoon_stars = star_nodes[2:]
    
    # Get all edges
    all_edges = list(G.edges())
    
    # Create specialized subgraphs for each star location
    morning_subgraphs = []
    afternoon_subgraphs = []
    
    # For each star location, find nearby edges
    for star in morning_stars:
        # Create a subgraph of edges within 2 hops of star
        neighbors = set([star])
        for _ in range(2):  # 2 hops
            new_neighbors = set()
            for n in neighbors:
                new_neighbors.update(G.neighbors(n))
            neighbors.update(new_neighbors)
        
        star_subgraph = G.subgraph(neighbors)
        morning_subgraphs.append(set(star_subgraph.edges()))
    
    for star in afternoon_stars:
        # Create a subgraph of edges within 2 hops of star
        neighbors = set([star])
        for _ in range(2):  # 2 hops
            new_neighbors = set()
            for n in neighbors:
                new_neighbors.update(G.neighbors(n))
            neighbors.update(new_neighbors)
        
        star_subgraph = G.subgraph(neighbors)
        afternoon_subgraphs.append(set(star_subgraph.edges()))
    
    # Combine star subgraphs
    morning_edges = set()
    for subgraph in morning_subgraphs:
        morning_edges.update(subgraph)
    
    afternoon_edges = set()
    for subgraph in afternoon_subgraphs:
        afternoon_edges.update(subgraph)
    
    # For remaining edges, split evenly
    remaining_edges = set(all_edges) - morning_edges - afternoon_edges
    
    # Count travel time for current assignments
    morning_time = sum(G[u][v]['time'] for u, v in morning_edges)
    afternoon_time = sum(G[u][v]['time'] for u, v in afternoon_edges)
    
    # Sort remaining edges by travel time (descending)
    remaining_edges = sorted(remaining_edges, key=lambda e: G[e[0]][e[1]]['time'], reverse=True)
    
    # Distribute remaining edges to balance time
    for u, v in remaining_edges:
        edge_time = G[u][v]['time']
        if morning_time <= afternoon_time:
            morning_edges.add((u, v))
            morning_time += edge_time
        else:
            afternoon_edges.add((u, v))
            afternoon_time += edge_time
    
    # Convert sets back to lists
    morning_edges = list(morning_edges)
    afternoon_edges = list(afternoon_edges)
    
    # Verify complete coverage
    all_edges_set = set(all_edges)
    covered_edges = set(morning_edges).union(set(afternoon_edges))
    
    if len(all_edges_set - covered_edges) > 0:
        print(f"Warning: {len(all_edges_set - covered_edges)} edges are not covered!")
        # Add any missing edges
        for edge in all_edges_set - covered_edges:
            if morning_time <= afternoon_time:
                morning_edges.append(edge)
                morning_time += G[edge[0]][edge[1]]['time']
            else:
                afternoon_edges.append(edge)
                afternoon_time += G[edge[0]][edge[1]]['time']
    
    print(f"Morning route: {len(morning_edges)} edges, estimated {morning_time:.1f} minutes")
    print(f"Afternoon route: {len(afternoon_edges)} edges, estimated {afternoon_time:.1f} minutes")
    
    return morning_edges, afternoon_edges, morning_stars, afternoon_stars

# Optimize route with TSP-like approach
def create_optimized_route(edges, star_nodes, G, start_node=None):
    """Create an optimized route that visits all edges and star nodes."""
    # Create a subgraph with only the selected edges
    subgraph = nx.Graph()
    for u, v in edges:
        if G.has_edge(u, v):
            subgraph.add_edge(u, v, **G.get_edge_data(u, v))
    
    # If no start node provided, use first star or any node
    if start_node is None:
        if star_nodes:
            # Find which star nodes are in this subgraph
            valid_stars = [star for star in star_nodes if star in subgraph]
            if valid_stars:
                start_node = valid_stars[0]
            else:
                start_node = list(subgraph.nodes())[0]
        else:
            start_node = list(subgraph.nodes())[0]
    
    # Ensure start node is in the graph
    if start_node not in subgraph:
        start_node = list(subgraph.nodes())[0]
    
    # Create a route that visits each edge exactly once if possible
    if nx.is_eulerian(subgraph):
        route = list(nx.eulerian_circuit(subgraph, source=start_node))
        print("Created Eulerian circuit")
    else:
        # Use a modified DFS traversal to maximize coverage
        route = []
        visited = set()
        
        def dfs_route(node):
            for neighbor in subgraph.neighbors(node):
                if (node, neighbor) not in visited and (neighbor, node) not in visited:
                    route.append((node, neighbor))
                    visited.add((node, neighbor))
                    visited.add((neighbor, node))
                    dfs_route(neighbor)
        
        dfs_route(start_node)
        
        # Check if all edges are covered
        all_edges = set(subgraph.edges())
        visited_edges = set((min(u, v), max(u, v)) for u, v in visited)
        missing_edges = all_edges - visited_edges
        
        if missing_edges:
            print(f"DFS traversal missed {len(missing_edges)} edges, adding them")
            # Try to connect missing edges to the existing route
            for u, v in missing_edges:
                if u in subgraph and v in subgraph:
                    route.append((u, v))
    
    # Optimize star visits: ensure stars are visited early in the route
    if star_nodes:
        # Find which star nodes are in this subgraph
        subgraph_stars = [star for star in star_nodes if star in subgraph]
        
        if subgraph_stars:
            # Check if stars are already in the route
            star_in_route = set()
            for u, v in route:
                if u in subgraph_stars:
                    star_in_route.add(u)
                if v in subgraph_stars:
                    star_in_route.add(v)
            
            # For stars not yet in route, try to insert them
            for star in subgraph_stars:
                if star not in star_in_route and star in subgraph:
                    # Find an edge connected to this star
                    for neighbor in subgraph.neighbors(star):
                        # Insert (star, neighbor) at the beginning of the route
                        route.insert(0, (star, neighbor))
                        star_in_route.add(star)
                        break
    
    return route

# Generate a time-constrained schedule with emphasis on star locations
def generate_schedule(route, G, star_nodes, speed=8.0, start_time_str='08:00', max_duration=240):
    """Generate a schedule within time constraints with emphasis on star locations."""
    original_route = route.copy()
    schedule = []
    visited_stars = set()
    
    # Parse start time
    start_time = datetime.strptime(start_time_str, '%H:%M')
    current_time = start_time
    end_time_limit = start_time + timedelta(minutes=max_duration)
    
    # First pass: Calculate travel times for each edge
    route_with_times = []
    for u, v in route:
        if G.has_edge(u, v):
            time_minutes = G[u][v]['time']
            route_with_times.append((u, v, time_minutes))
    
    # Second pass: Prioritize stars then organize by travel time
    # Reorganize route to prioritize star nodes
    prioritized_route = []
    star_edges = []
    
    # Find edges connected to star nodes
    for u, v, time in route_with_times:
        if u in star_nodes or v in star_nodes:
            star_edges.append((u, v, time))
        else:
            prioritized_route.append((u, v, time))
    
    # Put star edges at the beginning
    prioritized_route = star_edges + prioritized_route
    
    # Create schedule from prioritized route
    current_node = None
    
    for i, (u, v, travel_time) in enumerate(prioritized_route):
        # Skip if we've run out of time
        if current_time + timedelta(minutes=travel_time) > end_time_limit:
            continue
        
        # If we're not at the starting node of this edge, add a connector
        if current_node is not None and current_node != u:
            try:
                # Find shortest path in the graph
                path = nx.shortest_path(G, current_node, u, weight='time')
                
                # Add connector segments
                for j in range(len(path) - 1):
                    from_node, to_node = path[j], path[j + 1]
                    
                    if G.has_edge(from_node, to_node):
                        connector_time = G[from_node][to_node]['time']
                        connector_name = G[from_node][to_node]['name']
                        
                        # Check if adding connector exceeds time limit
                        if current_time + timedelta(minutes=connector_time) > end_time_limit:
                            break
                        
                        # Add to schedule
                        new_time = current_time + timedelta(minutes=connector_time)
                        schedule.append({
                            'from_node': from_node,
                            'to_node': to_node,
                            'road_name': f"Connector: {connector_name}",
                            'start_time': current_time.strftime('%H:%M'),
                            'end_time': new_time.strftime('%H:%M'),
                            'type': 'connector'
                        })
                        
                        current_time = new_time
            except nx.NetworkXNoPath:
                print(f"Warning: No path found from {current_node} to {u}")
                # If no path exists, just jump to the new node
        
        # Initialize current_node if this is the first edge
        if current_node is None:
            current_node = u
        
        # Add the edge to the schedule
        new_time = current_time + timedelta(minutes=travel_time)
        
        # Check if we're still within time limit
        if new_time > end_time_limit:
            break
        
        schedule.append({
            'from_node': u,
            'to_node': v,
            'road_name': G[u][v]['name'],
            'start_time': current_time.strftime('%H:%M'),
            'end_time': new_time.strftime('%H:%M'),
            'type': 'sampling'
        })
        
        current_time = new_time
        current_node = v
        
        # Check if we're at a star location
        if v in star_nodes and v not in visited_stars:
            visited_stars.add(v)
            
            # Add star stop
            star_start = current_time.strftime('%H:%M')
            star_end = current_time + timedelta(minutes=20)
            
            # Check if star stop exceeds time limit
            if star_end > end_time_limit:
                break
            
            schedule.append({
                'from_node': v,
                'to_node': v,
                'road_name': f"FIXED SAMPLING at location {v}",
                'start_time': star_start,
                'end_time': star_end.strftime('%H:%M'),
                'type': 'star_stop'
            })
            
            current_time = star_end
    
    # Calculate statistics
    if schedule:
        start = datetime.strptime(schedule[0]['start_time'], '%H:%M')
        end = datetime.strptime(schedule[-1]['end_time'], '%H:%M')
        duration = (end - start).total_seconds() / 60
    else:
        duration = 0
    
    # Calculate road coverage
    roads_covered = set()
    for item in schedule:
        if item['type'] in ['sampling', 'connector']:
            # Extract actual road name from connector prefix if needed
            name = item['road_name']
            if name.startswith("Connector: "):
                name = name[11:]  # Remove "Connector: " prefix
            roads_covered.add(name)
    
    # Report missing star locations
    missing_stars = set(star_nodes) - visited_stars
    if missing_stars:
        print(f"Warning: {len(missing_stars)} star locations not visited: {missing_stars}")
    
    print(f"Schedule covers {len(schedule)} segments, {len(roads_covered)} unique roads")
    print(f"Schedule duration: {duration:.1f} minutes with {len(visited_stars)} star locations visited")
    
    return schedule, duration, visited_stars, roads_covered

# Iterative improvement algorithm for star locations
def ensure_star_visits(G, route, star_nodes, time_limit=240):
    """Modify route to ensure all star locations are visited within time limit."""
    # Check which stars are already in the route
    stars_in_route = set()
    for u, v in route:
        if u in star_nodes:
            stars_in_route.add(u)
        if v in star_nodes:
            stars_in_route.add(v)
    
    # For stars not in route, insert them
    missing_stars = set(star_nodes) - stars_in_route
    
    if not missing_stars:
        return route  # All stars are already in the route
    
    # Create a new route with stars inserted at strategic positions
    new_route = []
    used_stars = set()
    
    # Start with a star node if possible
    if star_nodes and star_nodes[0] not in used_stars:
        star = star_nodes[0]
        neighbors = list(G.neighbors(star))
        if neighbors:
            new_route.append((star, neighbors[0]))
            used_stars.add(star)
    
    # Add existing route edges
    for u, v in route:
        new_route.append((u, v))
        if u in star_nodes:
            used_stars.add(u)
        if v in star_nodes:
            used_stars.add(v)
    
    # Try to insert remaining missing stars
    still_missing = set(star_nodes) - used_stars
    
    for star in still_missing:
        # Find a neighbor of this star
        neighbors = list(G.neighbors(star))
        if neighbors:
            # Insert at beginning to ensure it's visited
            new_route.insert(0, (star, neighbors[0]))
            used_stars.add(star)
    
    # Final check
    final_missing = set(star_nodes) - used_stars
    if final_missing:
        print(f"Warning: Could not insert {len(final_missing)} star locations into route")
    
    return new_route

# Visualize the route
def visualize_route(route, star_nodes, G, nodes_gdf, edges_gdf, star_locations, title):
    """Create a visual map of the sampling route."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all edges in light gray
    edges_gdf.plot(ax=ax, color='lightgray', linewidth=1, alpha=0.5)
    
    # Plot star locations
    star_locations.plot(ax=ax, color='red', markersize=100, marker='*')
    
    # Extract route edges for visualization
    route_edges = []
    for u, v in route:
        # Match with the GeoDataFrame
        edge_match = edges_gdf[(edges_gdf['u'] == u) & (edges_gdf['v'] == v)]
        if edge_match.empty:
            # Try reverse direction
            edge_match = edges_gdf[(edges_gdf['u'] == v) & (edges_gdf['v'] == u)]
        
        if not edge_match.empty:
            route_edges.append(edge_match)
    
    # Plot route if we found matches
    if route_edges:
        route_gdf = pd.concat(route_edges)
        route_gdf.plot(ax=ax, color='blue', linewidth=2)
    
    # Add markers for star nodes
    for i, star in enumerate(star_nodes):
        node_data = nodes_gdf[nodes_gdf['osmid'] == star]
        if not node_data.empty:
            x, y = node_data.iloc[0]['x'], node_data.iloc[0]['y']
            ax.scatter(x, y, color='orange', s=100, zorder=5)
            ax.annotate(f"Star {i+1}", (x, y), xytext=(5, 5), 
                       textcoords="offset points", fontsize=10, fontweight='bold')
    
    # Add title and legend
    plt.title(title)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Sampling Route'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Fixed Sampling Points')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.close()
    print(f"Saved visualization to {title.replace(' ', '_')}.png")

# Main function to design the sampling plan
def design_sampling_plan(file_path):
    """Design a comprehensive mobile air quality sampling plan."""
    # Load data
    nodes, edges, star_locations = load_road_network(file_path)
    
    # Create graph with realistic travel times
    G = create_graph(nodes, edges)
    
    # Find star nodes
    star_nodes = find_star_nodes(G, star_locations, nodes)
    
    # Split roads between morning and afternoon
    morning_edges, afternoon_edges, morning_stars, afternoon_stars = split_roads_balanced(G, star_nodes)
    
    # Create optimized routes
    morning_route = create_optimized_route(morning_edges, morning_stars, G)
    afternoon_route = create_optimized_route(afternoon_edges, afternoon_stars, G)
    
    # Ensure star visits
    morning_route = ensure_star_visits(G, morning_route, morning_stars)
    afternoon_route = ensure_star_visits(G, afternoon_route, afternoon_stars)
    
    # Generate schedules
    morning_schedule, morning_duration, morning_visited, morning_roads = generate_schedule(
        morning_route, G, morning_stars, speed=8.0, start_time_str='08:00', max_duration=240)
    
    afternoon_schedule, afternoon_duration, afternoon_visited, afternoon_roads = generate_schedule(
        afternoon_route, G, afternoon_stars, speed=8.0, start_time_str='13:00', max_duration=240)
    
    # Visualize routes
    visualize_route(morning_route, morning_stars, G, nodes, edges, star_locations, "Morning Sampling Plan")
    visualize_route(afternoon_route, afternoon_stars, G, nodes, edges, star_locations, "Afternoon Sampling Plan")
    
    return morning_schedule, afternoon_schedule, morning_duration, afternoon_duration, morning_roads, afternoon_roads

# Create a human-readable report
def create_report(morning_schedule, afternoon_schedule, morning_duration, afternoon_duration, 
                 morning_roads, afternoon_roads):
    """Create a human-readable report of the sampling plan."""
    with open("sampling_plan_report.txt", "w") as f:
        f.write("MOBILE AIR QUALITY SAMPLING PLAN\n")
        f.write("===============================\n\n")
        
        # Morning section
        f.write("MORNING SAMPLING PLAN (8:00 AM - 12:00 PM)\n")
        f.write(f"Total duration: {morning_duration:.1f} minutes\n")
        f.write(f"Roads covered: {len(morning_roads)}\n\n")
        
        for item in morning_schedule:
            if item['type'] == 'star_stop':
                f.write(f"* {item['start_time']} - {item['end_time']}: {item['road_name']}\n")
            else:
                f.write(f"  {item['start_time']} - {item['end_time']}: {item['road_name']}\n")
        
        f.write("\n")
        
        # Afternoon section
        f.write("AFTERNOON SAMPLING PLAN (1:00 PM - 5:00 PM)\n")
        f.write(f"Total duration: {afternoon_duration:.1f} minutes\n")
        f.write(f"Roads covered: {len(afternoon_roads)}\n\n")
        
        for item in afternoon_schedule:
            if item['type'] == 'star_stop':
                f.write(f"* {item['start_time']} - {item['end_time']}: {item['road_name']}\n")
            else:
                f.write(f"  {item['start_time']} - {item['end_time']}: {item['road_name']}\n")
        
        f.write("\n")
        
        # Summary section
        all_roads = morning_roads.union(afternoon_roads)
        f.write("SUMMARY\n")
        f.write("=======\n")
        f.write(f"Total unique roads covered: {len(all_roads)}\n")
        
        # Check for Drumm Avenue
        drumm_covered = any("Drumm" in road for road in all_roads)
        f.write(f"Drumm Avenue covered: {'Yes' if drumm_covered else 'No'}\n\n")
        
        # List all roads covered
        f.write("ROADS COVERED\n")
        f.write("=============\n")
        for road in sorted(all_roads):
            f.write(f"- {road}\n")
    
    print("Created sampling plan report: sampling_plan_report.txt")

# Run the program
if __name__ == "__main__":
    file_path = 'road_network_sandbox.gpkg'
    
    # Design the sampling plan
    morning_schedule, afternoon_schedule, morning_duration, afternoon_duration, morning_roads, afternoon_roads = design_sampling_plan(file_path)
    
    # Print summary
    print("\nSAMPLING PLAN SUMMARY:")
    print(f"Morning route: {len(morning_schedule)} segments, {morning_duration:.1f} minutes")
    print(f"Afternoon route: {len(afternoon_schedule)} segments, {afternoon_duration:.1f} minutes")
    print(f"Total roads covered: {len(morning_roads.union(afternoon_roads))}")
    
    # Export schedules to CSV
    pd.DataFrame(morning_schedule).to_csv('morning_schedule.csv', index=False)
    pd.DataFrame(afternoon_schedule).to_csv('afternoon_schedule.csv', index=False)
    
    # Create human-readable report
    create_report(morning_schedule, afternoon_schedule, morning_duration, afternoon_duration, morning_roads, afternoon_roads)