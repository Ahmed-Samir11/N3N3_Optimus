"""Debug Solver 75 assignment"""

from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_76 import dqn_assignment, get_dqn_network

env = LogisticsEnvironment()
dqn_net = get_dqn_network()

print("Testing DQN assignment...")
print(f"Total orders: {len(env.get_all_order_ids())}")
print(f"Total vehicles: {len(list(env.get_all_vehicles()))}")
print(f"Warehouses: {list(env.warehouses.keys())}")
print()

order_assignments, warehouse_inventory = dqn_assignment(env, dqn_net)

print(f"Assigned vehicles: {len(order_assignments)}")
for vehicle_id, orders in order_assignments.items():
    print(f"  {vehicle_id}: {len(orders)} orders")
    for oid, wh in orders.items():
        print(f"    - {oid} → {wh}")

if not order_assignments:
    print("\nERROR: No assignments made!")
    print("\nChecking first order manually...")
    order_id = env.get_all_order_ids()[0]
    requirements = env.get_order_requirements(order_id)
    print(f"Order: {order_id}")
    print(f"Requirements: {requirements}")
    
    for wh_id, wh in env.warehouses.items():
        inv = env.get_warehouse_inventory(wh_id)
        print(f"\nWarehouse {wh_id}:")
        print(f"  Inventory: {dict(inv)}")
        can_fulfill = all(inv.get(sku, 0) >= qty for sku, qty in requirements.items())
        print(f"  Can fulfill: {can_fulfill}")
        
        # Check vehicles
        vehicles = [v for v in env.get_all_vehicles() if v.home_warehouse_id == wh_id]
        print(f"  Vehicles: {len(vehicles)}")
        for v in vehicles:
            print(f"    - {v.id}")
            try:
                rem_w, rem_v = env.get_vehicle_remaining_capacity(v.id)
                print(f"      Capacity: {rem_w}kg, {rem_v}m³")
            except Exception as e:
                print(f"      Capacity error: {e}")
