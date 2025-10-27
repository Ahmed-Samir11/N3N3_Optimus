#!/usr/bin/env python3
"""Debug distance calculation."""

from robin_logistics import LogisticsEnvironment

env = LogisticsEnvironment()

# Check if get_distance is available
print("=" * 80)
print("DISTANCE FUNCTION DEBUG")
print("=" * 80)

# Get nodes
wh_node = list(env.warehouses.values())[0].location.id
order_node = env.orders[env.get_all_order_ids()[0]].destination.id

print(f"\nWarehouse node: {wh_node}")
print(f"Order node: {order_node}")

# Test different distance methods
print(f"\nTesting distance methods:")
print(f"  env.get_distance(): {env.get_distance(wh_node, order_node)}")

# Check if nodes exist
print(f"\n  Warehouse node in env.nodes: {wh_node in env.nodes}")
print(f"  Order node in env.nodes: {order_node in env.nodes}")

# Check node structure
if wh_node in env.nodes:
    wh_node_obj = env.nodes[wh_node]
    print(f"\nWarehouse node object: {wh_node_obj}")
    print(f"  Has lat: {hasattr(wh_node_obj, 'lat')}")
    print(f"  Has lon: {hasattr(wh_node_obj, 'lon')}")
    if hasattr(wh_node_obj, 'lat'):
        print(f"  Lat: {wh_node_obj.lat}, Lon: {wh_node_obj.lon}")

# Try getting route distance
print(f"\n  env.get_route_distance([{wh_node}, {order_node}]): {env.get_route_distance([wh_node, order_node])}")

# Check if road network is available
road_network = env.get_road_network_data()
print(f"\nRoad network available: {road_network is not None}")
if road_network:
    print(f"  Keys: {road_network.keys()}")

print("\n" + "=" * 80)
