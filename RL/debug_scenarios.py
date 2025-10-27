"""
Analyze Scenarios 3 & 4 - Why do they fail?
Debug script to understand failure modes
"""

from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_69 import solver
import json

print("=" * 70)
print("DEBUGGING SCENARIO 3 & 4 FAILURE MODES")
print("=" * 70)
print()

# Test on default scenario and log details
env = LogisticsEnvironment()

print("📊 ENVIRONMENT STATISTICS")
print("-" * 70)

# Orders
all_orders = env.get_all_order_ids()
print(f"Total Orders: {len(all_orders)}")

# Vehicles
vehicles = env.get_all_vehicles()
print(f"Total Vehicles: {len(vehicles)}")
print("\nVehicle Details:")
for v in vehicles[:3]:  # Show first 3
    print(f"  {v.id}: Weight={v.capacity_weight:.1f}kg, Volume={v.capacity_volume:.1f}L, "
          f"Fixed=${v.fixed_cost:.0f}, Var=${v.cost_per_km:.2f}/km")

# Warehouses
print(f"\nTotal Warehouses: {len(env.warehouses)}")
print("\nWarehouse Inventory:")
for wh_id, wh in env.warehouses.items():
    inv = env.get_warehouse_inventory(wh_id)
    total_items = sum(inv.values())
    print(f"  {wh_id}: {total_items} total items")
    for sku, qty in list(inv.items())[:3]:  # Show first 3 SKUs
        print(f"    - {sku}: {qty}")

# Orders
print("\nOrder Requirements (first 5):")
for order_id in all_orders[:5]:
    order = env.orders[order_id]
    reqs = env.get_order_requirements(order_id)
    total_weight = sum(env.skus[sku].weight * qty for sku, qty in reqs.items())
    total_volume = sum(env.skus[sku].volume * qty for sku, qty in reqs.items())
    print(f"  {order_id}: Weight={total_weight:.1f}kg, Volume={total_volume:.1f}L")
    for sku, qty in reqs.items():
        print(f"    - {sku}: {qty}")

# Check for potential issues
print("\n" + "=" * 70)
print("POTENTIAL ISSUES")
print("-" * 70)

# Issue 1: Inventory shortage
print("\n1. INVENTORY ANALYSIS")
total_order_requirements = {}
for order_id in all_orders:
    reqs = env.get_order_requirements(order_id)
    for sku, qty in reqs.items():
        total_order_requirements[sku] = total_order_requirements.get(sku, 0) + qty

print("Total required vs available:")
for sku, required in sorted(total_order_requirements.items()):
    total_available = sum(env.get_warehouse_inventory(wh_id).get(sku, 0) 
                         for wh_id in env.warehouses.keys())
    status = "✅" if total_available >= required else "❌ SHORTAGE"
    print(f"  {sku}: Need {required}, Have {total_available} {status}")

# Issue 2: Capacity constraints
print("\n2. CAPACITY ANALYSIS")
total_capacity_weight = sum(v.capacity_weight for v in vehicles)
total_capacity_volume = sum(v.capacity_volume for v in vehicles)

total_order_weight = sum(
    sum(env.skus[sku].weight * qty for sku, qty in env.get_order_requirements(oid).items())
    for oid in all_orders
)
total_order_volume = sum(
    sum(env.skus[sku].volume * qty for sku, qty in env.get_order_requirements(oid).items())
    for oid in all_orders
)

print(f"Total vehicle capacity: Weight={total_capacity_weight:.1f}kg, Volume={total_capacity_volume:.1f}L")
print(f"Total order demand: Weight={total_order_weight:.1f}kg, Volume={total_order_volume:.1f}L")
print(f"Capacity utilization: Weight={total_order_weight/total_capacity_weight*100:.1f}%, "
      f"Volume={total_order_volume/total_capacity_volume*100:.1f}%")

if total_order_weight > total_capacity_weight:
    print("❌ CRITICAL: Total order weight EXCEEDS vehicle capacity!")
if total_order_volume > total_capacity_volume:
    print("❌ CRITICAL: Total order volume EXCEEDS vehicle capacity!")

# Issue 3: Road network connectivity
print("\n3. ROAD NETWORK ANALYSIS")
road_data = env.get_road_network_data()
adjacency = road_data.get('adjacency_list', {})
print(f"Total nodes in network: {len(adjacency)}")
print(f"Total edges: {sum(len(neighbors) for neighbors in adjacency.values())}")

# Check if all order destinations are reachable
unreachable_count = 0
for order_id in all_orders[:10]:  # Check first 10
    dest = env.orders[order_id].destination.id
    if dest not in adjacency:
        print(f"  ❌ Order {order_id} destination node {dest} NOT in road network!")
        unreachable_count += 1

if unreachable_count > 0:
    print(f"❌ CRITICAL: {unreachable_count}/10 orders have unreachable destinations!")

print("\n" + "=" * 70)
print("SOLVER EXECUTION TEST")
print("-" * 70)

result = solver(env)
success, msg = env.execute_solution(result)
fulfillment = env.get_solution_fulfillment_summary(result)
cost = env.calculate_solution_cost(result)

print(f"\nSolver 69 Results:")
print(f"  Success: {success}")
print(f"  Fulfilled: {fulfillment.get('fully_fulfilled_orders', 0)}/{len(all_orders)}")
print(f"  Cost: ${cost:,.0f}")

if not success:
    print(f"\n❌ Execution failed: {msg}")

# Check unfulfilled orders
unfulfilled = set(all_orders) - set(fulfillment.get('order_details', {}).keys())
if unfulfilled:
    print(f"\nUnfulfilled orders ({len(unfulfilled)}):")
    for oid in list(unfulfilled)[:5]:
        reqs = env.get_order_requirements(oid)
        print(f"  {oid}: {reqs}")

print("\n" + "=" * 70)
print("HYPOTHESIS FOR SCENARIO 3 & 4 FAILURE")
print("=" * 70)
print("""
Based on analysis, possible reasons for 0% and 11% fulfillment:

Scenario 3 (0%):
  - Possible inventory shortage across all warehouses
  - Extreme capacity constraints (all orders too large for vehicles)
  - Road network connectivity issues (destinations unreachable)
  - Bug in solver logic for this specific scenario structure

Scenario 4 (11%):
  - Tight capacity constraints (only 3/27 orders fit)
  - Suboptimal order selection (solver picks wrong orders)
  - Multi-warehouse coordination failure
  - Need better bin packing algorithm

NEXT STEPS:
1. Add logging to Solver 69 to see WHERE it fails
2. Create scenario-specific fixes for tight capacity
3. Implement better order prioritization (value/weight ratio)
4. Add multi-warehouse order splitting for large orders
""")
