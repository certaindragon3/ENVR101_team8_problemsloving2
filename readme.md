# ENVR101 Team8 Problem Solving 2: Mobile Air Quality Sampling Plan - Drumm Avenue Neighborhood

## Background

The Drumm Avenue neighborhood in Wilmington, Los Angeles has been significantly impacted by increased truck traffic due to the permanent closure of railroad crossings between Lomita Boulevard and Alameda Street. This closure has transformed Drumm Avenue from a quiet residential road into a major truck route, severely affecting the quality of life for local residents through increased pollution, noise, and safety concerns.

Residents report respiratory problems, children are unable to play outside, and families must keep windows closed to mitigate noise and pollution. This computational project was developed to design an optimal mobile air quality sampling plan to scientifically document pollution levels, providing evidence to prompt governmental action and secure funding for mitigation measures.

## Problem Framing and Simplification

To make this complex routing problem algorithmically tractable, we implemented several strategic simplifications:

### Simplifying Constraints
1. **Fixed Vehicle Speed**: We set a constant speed of 8 m/s (28.8 km/h) throughout the sampling process, balancing between the suggested optimal speed (5 m/s) and the maximum allowable speed (10 m/s). This simplification makes time calculations more reliable.

2. **Exact Time Windows**: Routes are designed to fit precisely within the available 4-hour windows (8:00 AM - 12:00 PM and 1:00 PM - 5:00 PM), maximizing data collection efficiency.

3. **Practical Road Network**: While aiming for complete road coverage, we prioritized creating cohesive routes that minimize fragmentation and unnecessary backtracking.

### Digital Sandbox Creation
1. The `sandbox_creation.py` script extracts road network data from OpenStreetMap and creates a GeoPackage (GPKG) file containing accurate road geometries, intersections, and distances for the study area.

2. After generating the base road network, we used QGIS to precisely place the four fixed sampling locations (marked as stars) based on their geographic coordinates within the study area.

## Reproduction Manual

### Prerequisites
- Python 3.8 or higher
- Required packages:
  - osmnx
  - geopandas
  - networkx
  - matplotlib
  - numpy
  - pandas
  - shapely

### Installation
```bash
# Clone the repository
git clone https://github.com/certaindragon3/ENVR101_team8_problemsloving2.git
cd ENVR101_team8_problemsloving2

# Install required dependencies
pip install -r requirements.txt
```

### Running the Code
1. **Create the Road Network Sandbox** (Optional - already included in the repository):
```bash
python sandbox_creation.py
```
This will generate a `road_network_sandbox.gpkg` file containing the road network data.

**Note**: The repository already contains the `road_network_sandbox.gpkg` file with the star locations properly added, so you can skip the sandbox creation and star location steps if you want to directly generate the sampling plan.

2. **Generate the Sampling Plan**:
```bash
python algorithm.py
```
This will create the optimized sampling routes and generate results in the `output` directory.

## Code Design Highlights

The core algorithm implements several sophisticated techniques:

1. **Balanced Road Allocation**: Divides roads between morning and afternoon sessions to ensure complete coverage.
```python
def split_roads_balanced(G, star_nodes):
    # Split star nodes for morning and afternoon
    morning_stars = star_nodes[:2]
    afternoon_stars = star_nodes[2:]
    
    # Distribute edges to balance travel time
    for u, v in remaining_edges:
        edge_time = G[u][v]['time']
        if morning_time <= afternoon_time:
            morning_edges.add((u, v))
            morning_time += edge_time
        else:
            afternoon_edges.add((u, v))
            afternoon_time += edge_time
```

2. **Star Location Priority**: Guarantees all four fixed sampling locations are visited with their required 20-minute stops.
```python
def ensure_star_visits(G, route, star_nodes, time_limit=240):
    # For stars not in route, insert them
    missing_stars = set(star_nodes) - stars_in_route
    
    for star in still_missing:
        # Find a neighbor of this star
        neighbors = list(G.neighbors(star))
        if neighbors:
            # Insert at beginning to ensure it's visited
            new_route.insert(0, (star, neighbors[0]))
            used_stars.add(star)
```

3. **Time-Constrained Routing**: Creates routes that exactly fit the 4-hour time windows.
```python
def generate_schedule(route, G, star_nodes, speed=8.0, start_time_str='08:00', max_duration=240):
    # Parse start time
    start_time = datetime.strptime(start_time_str, '%H:%M')
    current_time = start_time
    end_time_limit = start_time + timedelta(minutes=max_duration)
    
    # Add the edge to the schedule
    new_time = current_time + timedelta(minutes=travel_time)
    
    # Check if we're still within time limit
    if new_time > end_time_limit:
        break
```

4. **Connected Path Generation**: Uses graph theory to ensure routes are practical to follow with minimal fragmentation.
```python
def create_optimized_route(edges, star_nodes, G, start_node=None):
    # Create a route that visits each edge exactly once if possible
    if nx.is_eulerian(subgraph):
        route = list(nx.eulerian_circuit(subgraph, source=start_node))
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
```

5. **Visualization**: Creates clear maps of the routes for easy reference.
```python
def visualize_route(route, star_nodes, G, nodes_gdf, edges_gdf, star_locations, title):
    # Plot all edges in light gray
    edges_gdf.plot(ax=ax, color='lightgray', linewidth=1, alpha=0.5)
    
    # Plot star locations
    star_locations.plot(ax=ax, color='red', markersize=100, marker='*')
    
    # Plot route
    if route_edges:
        route_gdf = pd.concat(route_edges)
        route_gdf.plot(ax=ax, color='blue', linewidth=2)
```

## Output Files

The algorithm generates several output files in the `output` directory:

1. **Visual Route Maps** (`Morning_Sampling_Plan.png`, `Afternoon_Sampling_Plan.png`):
   - Blue lines represent the sampling routes
   - Red stars indicate fixed sampling locations

2. **Schedule CSV Files** (`morning_schedule.csv`, `afternoon_schedule.csv`):
   - Contains detailed timestamped routes with road names
   - Indicates sampling mode (sampling/connector/star_stop)
   - Provides start and end times for each segment

3. **Comprehensive Report** (`sampling_plan_report.txt`):
   - Complete summary of the sampling plan
   - Lists all roads covered
   - Provides timing details for all segments
   - Highlights fixed sampling location visits

## Project Outcome

The complete analysis and detailed sampling plan results are available in the report PDF. Our computational approach provides an efficient solution that meets all requirements:

- Visits all 18 roads within the sampling area at least once daily
- Properly allocates routes between morning and afternoon sessions
- Ensures all four fixed sampling locations receive their required 20-minute stops
- Fits perfectly within the specified time windows
- Balances between coverage needs and time constraints

This plan effectively addresses the key concerns while providing a practical solution for data collection to document air quality issues in the Drumm Avenue neighborhood.