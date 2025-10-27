"""
Calculate the theoretical minimum cost for a logistics scenario.
This represents the absolute best case - impossible to beat in practice.
"""

from robin_logistics import LogisticsEnvironment
import numpy as np
import networkx as nx

def calculate_theoretical_minimum(env):
    """
    Calculate theoretical minimum cost assuming:
    1. Perfect vehicle packing (100% utilization)
    2. Direct paths (no detours)
    3. Single warehouse per order (no splits)
    4. Optimal warehouse selection (closest to delivery)
    5. Minimal vehicle count
    """
    
    # Get road network
    network_data = env.get_road_network_data()
    G = nx.DiGraph()
    for edge in network_data['edges']:
        G.add_edge(edge['from'], edge['to'], weight=edge['distance'])
    
    print("="*80)
    print("THEORETICAL MINIMUM COST ANALYSIS")
    print("="*80)
    
    # 1. Calculate minimum vehicles needed
    total_weight = 0
    total_volume = 0
    
    for order_id in env.get_all_order_ids():
        reqs = env.get_order_requirements(order_id)
        for sku_id, qty in reqs.items():
            sku = env.skus[sku_id]
            total_weight += sku.weight * qty
            total_volume += sku.volume * qty
    
    # Get vehicle capacities (assume all same)
    sample_vehicle = list(env.get_all_vehicles())[0]
    vehicle_weight_cap = sample_vehicle.capacity_weight
    vehicle_volume_cap = sample_vehicle.capacity_volume
    vehicle_fixed_cost = sample_vehicle.fixed_cost
    cost_per_km = sample_vehicle.cost_per_km
    
    min_vehicles_by_weight = np.ceil(total_weight / vehicle_weight_cap)
    min_vehicles_by_volume = np.ceil(total_volume / vehicle_volume_cap)
    min_vehicles = int(max(min_vehicles_by_weight, min_vehicles_by_volume))
    
    print(f"\n1. MINIMUM VEHICLES NEEDED:")
    print(f"   Total weight: {total_weight:.2f} kg")
    print(f"   Total volume: {total_volume:.2f} m³")
    print(f"   Vehicle capacity: {vehicle_weight_cap:.2f} kg, {vehicle_volume_cap:.2f} m³")
    print(f"   Min vehicles (by weight): {min_vehicles_by_weight:.0f}")
    print(f"   Min vehicles (by volume): {min_vehicles_by_volume:.0f}")
    print(f"   **THEORETICAL MIN VEHICLES: {min_vehicles}**")
    print(f"   Fixed cost per vehicle: ${vehicle_fixed_cost:.2f}")
    print(f"   **MIN FIXED COST: ${min_vehicles * vehicle_fixed_cost:.2f}**")
    
    # 2. Calculate minimum distance (closest warehouse to each order)
    min_distance = 0
    warehouse_selections = {}
    
    for order_id in env.get_all_order_ids():
        order = env.orders[order_id]
        order_dest = order.destination.id
        
        best_distance = float('inf')
        best_warehouse = None
        
        for wh_id, warehouse in env.warehouses.items():
            wh_node = warehouse.location.id
            try:
                distance = nx.shortest_path_length(G, wh_node, order_dest, weight='weight')
                if distance < best_distance:
                    best_distance = distance
                    best_warehouse = wh_id
            except:
                pass
        
        if best_warehouse:
            min_distance += best_distance
            warehouse_selections[order_id] = (best_warehouse, best_distance)
    
    print(f"\n2. MINIMUM DISTANCE (single closest warehouse per order):")
    print(f"   Total orders: {len(env.get_all_order_ids())}")
    print(f"   **TOTAL MIN DISTANCE: {min_distance:.2f} km**")
    print(f"   Average distance per order: {min_distance / len(env.get_all_order_ids()):.2f} km")
    
    # 3. Calculate minimum variable cost
    min_variable_cost = min_distance * cost_per_km
    
    print(f"\n3. MINIMUM VARIABLE COST:")
    print(f"   Cost per km: ${cost_per_km:.2f}")
    print(f"   **MIN VARIABLE COST: ${min_variable_cost:.2f}**")
    
    # 4. Calculate theoretical minimum total cost
    min_total_cost = (min_vehicles * vehicle_fixed_cost) + min_variable_cost
    
    print(f"\n{'='*80}")
    print(f"THEORETICAL MINIMUM TOTAL COST: ${min_total_cost:.2f}")
    print(f"{'='*80}")
    print(f"   Fixed cost:    ${min_vehicles * vehicle_fixed_cost:.2f} ({(min_vehicles * vehicle_fixed_cost / min_total_cost * 100):.1f}%)")
    print(f"   Variable cost: ${min_variable_cost:.2f} ({(min_variable_cost / min_total_cost * 100):.1f}%)")
    print(f"{'='*80}\n")
    
    # 5. Analyze warehouse usage
    print(f"4. WAREHOUSE USAGE ANALYSIS:")
    wh_usage = {}
    for order_id, (wh_id, dist) in warehouse_selections.items():
        if wh_id not in wh_usage:
            wh_usage[wh_id] = {'count': 0, 'total_dist': 0}
        wh_usage[wh_id]['count'] += 1
        wh_usage[wh_id]['total_dist'] += dist
    
    for wh_id, stats in wh_usage.items():
        print(f"   {wh_id}: {stats['count']} orders, avg dist {stats['total_dist']/stats['count']:.2f} km")
    
    # 6. Check inventory constraints
    print(f"\n5. INVENTORY FEASIBILITY CHECK:")
    inventory_needed = {}
    for order_id in env.get_all_order_ids():
        if order_id not in warehouse_selections:
            print(f"   ⚠️ {order_id} has no reachable warehouse - skipping")
            continue
            
        wh_id = warehouse_selections[order_id][0]
        reqs = env.get_order_requirements(order_id)
        
        if wh_id not in inventory_needed:
            inventory_needed[wh_id] = {}
        
        for sku_id, qty in reqs.items():
            if sku_id not in inventory_needed[wh_id]:
                inventory_needed[wh_id][sku_id] = 0
            inventory_needed[wh_id][sku_id] += qty
    
    inventory_feasible = True
    for wh_id, needed in inventory_needed.items():
        available = env.get_warehouse_inventory(wh_id)
        for sku_id, qty_needed in needed.items():
            qty_available = available.get(sku_id, 0)
            if qty_needed > qty_available:
                print(f"   ⚠️ {wh_id} needs {qty_needed} {sku_id} but only has {qty_available}")
                inventory_feasible = False
    
    if inventory_feasible:
        print(f"   ✅ All warehouses have sufficient inventory!")
    else:
        print(f"   ❌ Some warehouses lack inventory - multi-warehouse splits required")
        print(f"   NOTE: Theoretical minimum assumes perfect inventory - real cost will be higher")
    
    return {
        'min_vehicles': min_vehicles,
        'min_fixed_cost': min_vehicles * vehicle_fixed_cost,
        'min_distance': min_distance,
        'min_variable_cost': min_variable_cost,
        'min_total_cost': min_total_cost,
        'warehouse_selections': warehouse_selections,
        'inventory_feasible': inventory_feasible
    }

def compare_with_actual(env, solver_func, solver_name):
    """Compare actual solver performance with theoretical minimum"""
    
    print(f"\n{'='*80}")
    print(f"COMPARING {solver_name.upper()} WITH THEORETICAL MINIMUM")
    print(f"{'='*80}\n")
    
    # Get theoretical minimum
    theoretical = calculate_theoretical_minimum(env)
    
    # Run actual solver
    print(f"\nRunning {solver_name}...")
    result = solver_func(env)
    
    success, msg = env.execute_solution(result)
    if not success:
        print(f"❌ Solver failed: {msg}")
        return
    
    fulfillment = env.get_solution_fulfillment_summary(result)
    actual_cost = env.calculate_solution_cost(result)
    
    vehicles_used = sum(1 for v_route in result["routes"] if len(v_route["steps"]) > 2)
    fulfilled = fulfillment.get("fully_fulfilled_orders", 0)
    total_orders = len(env.get_all_order_ids())
    
    print(f"\n{'='*80}")
    print(f"ACTUAL VS THEORETICAL COMPARISON")
    print(f"{'='*80}")
    print(f"\nVehicles:")
    print(f"   Theoretical min: {theoretical['min_vehicles']}")
    print(f"   Actual used:     {vehicles_used}")
    print(f"   Gap:             +{vehicles_used - theoretical['min_vehicles']} vehicles")
    
    print(f"\nCost:")
    print(f"   Theoretical min: ${theoretical['min_total_cost']:.2f}")
    print(f"   Actual cost:     ${actual_cost:.2f}")
    print(f"   Gap:             ${actual_cost - theoretical['min_total_cost']:.2f} (+{(actual_cost / theoretical['min_total_cost'] - 1) * 100:.1f}%)")
    
    print(f"\nFulfillment:")
    print(f"   Orders fulfilled: {fulfilled}/{total_orders} ({fulfilled/total_orders*100:.1f}%)")
    
    # Calculate optimization potential
    potential_savings = actual_cost - theoretical['min_total_cost']
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION POTENTIAL: ${potential_savings:.2f} ({potential_savings/actual_cost*100:.1f}%)")
    print(f"{'='*80}")
    
    # Breakdown of where savings could come from
    vehicle_gap = vehicles_used - theoretical['min_vehicles']
    if vehicle_gap > 0:
        vehicle_waste = vehicle_gap * (theoretical['min_fixed_cost'] / theoretical['min_vehicles'])
        print(f"\nPotential savings from vehicle optimization: ${vehicle_waste:.2f}")
    
    distance_waste = potential_savings - (vehicle_gap * (theoretical['min_fixed_cost'] / theoretical['min_vehicles']) if vehicle_gap > 0 else 0)
    print(f"Potential savings from route optimization: ${distance_waste:.2f}")
    
    print(f"\n{'='*80}\n")
    
    return {
        'theoretical': theoretical,
        'actual_cost': actual_cost,
        'vehicles_used': vehicles_used,
        'potential_savings': potential_savings
    }

if __name__ == "__main__":
    import random
    import sys
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    env = LogisticsEnvironment()
    
    # Calculate theoretical minimum
    theoretical = calculate_theoretical_minimum(env)
    
    # Optionally compare with solver
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        from Ne3Na3_solver_84 import solver as solver84
        compare_with_actual(env, solver84, "Solver 84")
