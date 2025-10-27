"""
Test solver 77 with detailed unfulfilled order analysis
"""

from Ne3Na3_solver_84 import solver
from robin_logistics import LogisticsEnvironment
import time

print("=" * 80)
print("SOLVER 77 - UNFULFILLED ORDER ANALYSIS")
print("=" * 80)

env = LogisticsEnvironment()
result = solver(env, debug=False)

# Get fulfillment details
fulfillment = env.get_solution_fulfillment_summary(result)
all_orders = set(env.get_all_order_ids())
fulfilled_orders = set(fulfillment.get('fully_fulfilled_order_ids', []))
unfulfilled_orders = all_orders - fulfilled_orders

print(f"\n{'='*80}")
print(f"FULFILLMENT SUMMARY")
print(f"{'='*80}")
print(f"Total orders: {len(all_orders)}")
print(f"Fulfilled: {len(fulfilled_orders)} ({100*len(fulfilled_orders)/len(all_orders):.1f}%)")
print(f"Unfulfilled: {len(unfulfilled_orders)} ({100*len(unfulfilled_orders)/len(all_orders):.1f}%)")

if unfulfilled_orders:
    print(f"\n{'='*80}")
    print(f"UNFULFILLED ORDER DETAILS")
    print(f"{'='*80}")
    
    for order_id in sorted(unfulfilled_orders):
        order = env.orders[order_id]
        reqs = env.get_order_requirements(order_id)
        
        print(f"\n{order_id}:")
        print(f"  Destination: Node {order.destination.id} ({order.destination.lat:.4f}, {order.destination.lon:.4f})")
        print(f"  Requirements:")
        
        total_weight = 0
        total_volume = 0
        for sku_id, qty in reqs.items():
            sku = env.skus[sku_id]
            weight = sku.weight * qty
            volume = sku.volume * qty
            total_weight += weight
            total_volume += volume
            print(f"    - {sku_id}: {qty} units ({weight:.1f}kg, {volume:.1f}m³)")
        
        print(f"  Total: {total_weight:.1f}kg, {total_volume:.1f}m³")
        
        # Check warehouse inventory
        print(f"  Warehouse inventory availability:")
        for wh_id, wh in env.warehouses.items():
            inv = env.get_warehouse_inventory(wh_id)
            has_all = all(inv.get(sku, 0) >= qty for sku, qty in reqs.items())
            if has_all:
                print(f"    ✓ {wh_id}: All SKUs available")
            else:
                missing = []
                for sku, qty in reqs.items():
                    available = inv.get(sku, 0)
                    if available < qty:
                        missing.append(f"{sku} (need {qty}, have {available})")
                print(f"    ✗ {wh_id}: Missing {', '.join(missing)}")
        
        # Check which vehicles could fit this order
        print(f"  Vehicles with sufficient capacity:")
        compatible_vehicles = []
        for vehicle in env.get_all_vehicles():
            if vehicle.capacity_weight >= total_weight and vehicle.capacity_volume >= total_volume:
                rem_weight, rem_volume = env.get_vehicle_remaining_capacity(vehicle.id)
                if rem_weight >= total_weight and rem_volume >= total_volume:
                    compatible_vehicles.append(vehicle.id)
        
        if compatible_vehicles:
            print(f"    Available: {', '.join(compatible_vehicles)}")
        else:
            print(f"    None - all vehicles at capacity or order too large")

print(f"\n{'='*80}")

# Check vehicle utilization
print(f"\n{'='*80}")
print(f"VEHICLE UTILIZATION")
print(f"{'='*80}")

for vehicle in env.get_all_vehicles():
    rem_weight, rem_volume = env.get_vehicle_remaining_capacity(vehicle.id)
    used_weight = vehicle.capacity_weight - rem_weight
    used_volume = vehicle.capacity_volume - rem_volume
    weight_pct = 100 * used_weight / vehicle.capacity_weight
    volume_pct = 100 * used_volume / vehicle.capacity_volume
    
    # Check if vehicle has orders
    has_orders = False
    for route in result.get('routes', []):
        if route['vehicle_id'] == vehicle.id:
            orders_in_route = []
            for step in route['steps']:
                orders_in_route.extend([d['order_id'] for d in step.get('deliveries', [])])
            if orders_in_route:
                has_orders = True
                print(f"\n{vehicle.id}:")
                print(f"  Capacity: {vehicle.capacity_weight}kg, {vehicle.capacity_volume}m³")
                print(f"  Used: {used_weight:.1f}kg ({weight_pct:.1f}%), {used_volume:.1f}m³ ({volume_pct:.1f}%)")
                print(f"  Remaining: {rem_weight:.1f}kg, {rem_volume:.1f}m³")
                print(f"  Orders: {', '.join(set(orders_in_route))}")
                break
    
    if not has_orders:
        print(f"\n{vehicle.id}: UNUSED")

print(f"\n{'='*80}")
