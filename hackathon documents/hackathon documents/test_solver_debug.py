#!/usr/bin/env python3
"""Debug script to check solver logic."""

from robin_logistics import LogisticsEnvironment

env = LogisticsEnvironment()

# Check environment structure
print("=" * 80)
print("ENVIRONMENT INSPECTION")
print("=" * 80)

print(f"\nNumber of orders: {len(env.get_all_order_ids())}")
print(f"Number of warehouses: {len(env.warehouses)}")
print(f"Number of vehicles: {len(env.get_all_vehicles())}")

# Check first order
order_ids = env.get_all_order_ids()
if order_ids:
    first_order_id = order_ids[0]
    first_order = env.orders[first_order_id]
    print(f"\nFirst order: {first_order_id}")
    print(f"  Destination node: {first_order.destination.id}")
    print(f"  Items: {first_order.requested_items}")

# Check first warehouse
for wh_id, warehouse in list(env.warehouses.items())[:1]:
    print(f"\nFirst warehouse: {wh_id}")
    print(f"  Location node: {warehouse.location.id}")
    print(f"  Inventory: {warehouse.inventory}")
    print(f"  Vehicles: {len(warehouse.vehicles)}")

# Test distance function
if order_ids:
    wh_node = list(env.warehouses.values())[0].location.id
    order_node = env.orders[order_ids[0]].destination.id
    dist = env.get_distance(wh_node, order_node)
    print(f"\nDistance from warehouse to first order: {dist}")

print("\n" + "=" * 80)
