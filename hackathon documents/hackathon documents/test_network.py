#!/usr/bin/env python3
"""Debug road network structure."""

from robin_logistics import LogisticsEnvironment

env = LogisticsEnvironment()

road_network = env.get_road_network_data()

print("Road Network Structure:")
print(f"Keys: {road_network.keys()}")
print(f"\nAdjacency list type: {type(road_network['adjacency_list'])}")
print(f"Edges type: {type(road_network['edges'])}")

# Sample adjacency
adj_list = road_network['adjacency_list']
sample_node = list(adj_list.keys())[0]
print(f"\nSample node: {sample_node}")
print(f"Neighbors: {adj_list[sample_node][:5]}")  # First 5 neighbors

# Sample edges
edges = road_network['edges']
print(f"\nTotal edges: {len(edges)}")
print(f"Sample edge: {edges[0]}")
