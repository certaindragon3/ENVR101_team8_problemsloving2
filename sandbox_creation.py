import osmnx as ox
from shapely.geometry import Polygon
import geopandas as gpd

# 1. 定义区域
coords = [[-118.2514479997, 33.7831484436], [-118.2411256067, 33.7831484436], [-118.2411256067, 33.7960575406], [-118.2514479997, 33.7960575406], [-118.2514479997, 33.7831484436]]
polygon = Polygon(coords)
network_type = 'drive'

# 输出文件名
output_geopackage = 'road_network_sandbox.gpkg'

try:
    # 2. 获取街道网络
    print(f"Fetching '{network_type}' network for the specified polygon...")
    G = ox.graph_from_polygon(polygon, network_type=network_type, simplify=True)
    print(f"Graph contains {len(G.nodes)} nodes and {len(G.edges)} edges.")

    # 3. 转换为 GeoDataFrames
    print("Converting graph to GeoDataFrames...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # --- FIX: Reset index earlier to access 'osmid' as a column ---
    # The node OSM ID is the index, reset it to become a column named 'osmid' (usually)
    nodes_gdf.reset_index(inplace=True)
    # Reset edge index (MultiIndex: u, v, key) to columns
    edges_gdf.reset_index(inplace=True)
    # -------------------------------------------------------------

    # 确保包含长度信息 (osmnx 通常会自动添加)
    if 'length' not in edges_gdf.columns:
        print("Warning: 'length' column not found, calculating edge lengths...")
        # Re-project graph if length needs calculation (shouldn't be needed if simplify=True worked)
        G_proj = ox.project_graph(G)
        # Recalculate edges_proj_gdf based on G_proj
        _, edges_proj_gdf = ox.graph_to_gdfs(G_proj)
        edges_proj_gdf.reset_index(inplace=True) # Reset index here too if recalculated
        # A robust way to merge length requires matching u, v, key which can be complex.
        # Assuming simplify=True added length reliably. If not, this part needs careful merging.
        # For simplicity, let's assume length was added correctly by ox.graph_to_gdfs initially.
        # If length calculation is needed, a more robust merge based on u,v,key would be required here.
        print("Note: Length calculation after initial fetch might require careful merging. Assuming initial length is present.")


    # 打印一些信息 (Now 'osmid' should be a column in nodes_gdf)
    print("\nSample Edges (Roads) Data:")
    # Display relevant edge columns if they exist
    edge_cols_to_show = ['u', 'v', 'key', 'osmid', 'name', 'highway', 'length', 'geometry']
    valid_edge_cols = [col for col in edge_cols_to_show if col in edges_gdf.columns]
    print(edges_gdf[valid_edge_cols].head())


    print("\nSample Nodes (Intersections) Data:")
    # Display relevant node columns if they exist
    node_cols_to_show = ['osmid', 'y', 'x', 'geometry']
    valid_node_cols = [col for col in node_cols_to_show if col in nodes_gdf.columns]
    print(nodes_gdf[valid_node_cols].head())


    # 4. 保存为 GeoPackage (沙盒文件)
    print(f"\nSaving nodes and edges to {output_geopackage}...")

    # 保存边 (道路) - 指定图层名称为 'edges'
    # Ensure geometry column exists before saving
    if 'geometry' in edges_gdf.columns:
        edges_gdf.to_file(output_geopackage, layer='edges', driver='GPKG')
    else:
        print("Error: 'geometry' column not found in edges GeoDataFrame. Cannot save edges.")


    # 保存节点 (交叉口) - 指定图层名称为 'nodes'
    # Ensure geometry column exists before saving
    if 'geometry' in nodes_gdf.columns:
         # Overwrite the file or use append=True if adding to existing file.
         # For simplicity, let's overwrite by saving nodes second.
         # A better approach for multiple layers is context manager or specific OGR options if needed.
        nodes_gdf.to_file(output_geopackage, layer='nodes', driver='GPKG')
    else:
        print("Error: 'geometry' column not found in nodes GeoDataFrame. Cannot save nodes.")


    # Check if both layers were attempted to be saved successfully (basic check)
    if 'geometry' in edges_gdf.columns and 'geometry' in nodes_gdf.columns:
        print("Data saved successfully (nodes and edges).")
        print(f"You can now open '{output_geopackage}' in GIS software (like QGIS)")
        print("to view the network and manually add your star locations.")

except Exception as e:
    print(f"An error occurred: {e}")