"""
Deep Route Validation Debugger
Identifies EXACT issues in route structure
"""

from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_69 import solver
import json

def inspect_route_details(env, solution):
    """Deep inspection of route structure"""
    
    routes = solution.get('routes', [])
    
    print("=" * 80)
    print("DETAILED ROUTE INSPECTION")
    print("=" * 80)
    
    for i, route in enumerate(routes[:3], 1):  # Inspect first 3 routes
        vehicle_id = route['vehicle_id']
        steps = route['steps']
        
        print(f"\n{'='*80}")
        print(f"ROUTE {i}: Vehicle {vehicle_id}")
        print(f"{'='*80}")
        print(f"Total steps: {len(steps)}")
        
        # Get vehicle info
        vehicle = None
        for v in env.get_all_vehicles():
            if v.id == vehicle_id:
                vehicle = v
                break
        
        if vehicle:
            home_wh = env.warehouses[vehicle.home_warehouse_id]
            home_node = home_wh.location.id
            print(f"Vehicle home warehouse: {vehicle.home_warehouse_id}")
            print(f"Expected home node: {home_node}")
        
        # Inspect each step
        print(f"\n{'Step':<6} {'Node ID':<10} {'Pickups':<10} {'Deliveries':<12} {'Issues':<40}")
        print("-" * 80)
        
        issues_found = []
        pickup_skus = set()  # Track what we've picked up
        
        for step_idx, step in enumerate(steps):
            node_id = step['node_id']
            pickups = step.get('pickups', [])
            deliveries = step.get('deliveries', [])
            
            # Format for display
            pickup_str = f"{len(pickups)} items" if pickups else "-"
            delivery_str = f"{len(deliveries)} items" if deliveries else "-"
            
            step_issues = []
            
            # CHECK 1: First step should be home node
            if step_idx == 0:
                if node_id != home_node:
                    step_issues.append(f"Wrong start (expect {home_node})")
            
            # CHECK 2: Last step should be home node
            if step_idx == len(steps) - 1:
                if node_id != home_node:
                    step_issues.append(f"Wrong end (expect {home_node})")
            
            # CHECK 3: Pickups should be at warehouse
            if pickups:
                warehouse_nodes = {wh.location.id: wh_id for wh_id, wh in env.warehouses.items()}
                if node_id not in warehouse_nodes:
                    step_issues.append(f"Pickup at non-warehouse node")
                else:
                    # Track picked up SKUs
                    for pickup in pickups:
                        sku = pickup.get('sku_id')
                        qty = pickup.get('quantity', 0)
                        pickup_skus.add(sku)
            
            # CHECK 4: Deliveries should only use picked-up SKUs
            if deliveries:
                for delivery in deliveries:
                    sku = delivery.get('sku_id')
                    if sku not in pickup_skus:
                        step_issues.append(f"Delivery of {sku} before pickup")
            
            # CHECK 5: Check if node exists in network
            road_data = env.get_road_network_data()
            adjacency = road_data.get('adjacency_list', {})
            if node_id not in adjacency:
                step_issues.append(f"Node {node_id} not in network")
            
            issue_str = "; ".join(step_issues) if step_issues else "OK"
            if step_issues:
                issues_found.extend(step_issues)
            
            print(f"{step_idx:<6} {node_id:<10} {pickup_str:<10} {delivery_str:<12} {issue_str:<40}")
        
        # Summary for this route
        print("\n" + "-" * 80)
        if issues_found:
            print(f"[X] Route {i} has {len(issues_found)} issues:")
            for issue in set(issues_found):
                print(f"   - {issue}")
        else:
            print(f"[OK] Route {i} structure looks valid")
        
        # Validate through API
        print("\nAPI Validation:")
        try:
            validation = env.validate_solution_complete(solution)
            if isinstance(validation, tuple) and len(validation) == 2:
                is_valid, msg = validation
                print(f"   Valid: {is_valid}")
                if not is_valid:
                    print(f"   Error: {msg}")
            else:
                print(f"   Result: {validation}")
        except Exception as e:
            print(f"   Error during validation: {e}")

def compare_pickup_delivery_order(env, solution):
    """Check if pickups happen before deliveries for each order"""
    
    print("\n" + "=" * 80)
    print("PICKUP vs DELIVERY SEQUENCE CHECK")
    print("=" * 80)
    
    routes = solution.get('routes', [])
    all_issues = []
    
    for route_idx, route in enumerate(routes):
        vehicle_id = route['vehicle_id']
        steps = route['steps']
        
        # Track order flow
        order_pickups = {}  # {order_id: step_index where we picked up its SKUs}
        order_deliveries = {}  # {order_id: step_index where we delivered}
        
        for step_idx, step in enumerate(steps):
            pickups = step.get('pickups', [])
            deliveries = step.get('deliveries', [])
            
            # Track deliveries
            for delivery in deliveries:
                order_id = delivery.get('order_id')
                if order_id:
                    order_deliveries[order_id] = step_idx
            
            # Track pickups (we need to match SKUs to orders)
            for pickup in pickups:
                warehouse_id = pickup.get('warehouse_id')
                sku_id = pickup.get('sku_id')
                qty = pickup.get('quantity')
                
                # Find which orders need this SKU
                for order_id in env.get_all_order_ids():
                    reqs = env.get_order_requirements(order_id)
                    if sku_id in reqs and order_id in order_deliveries:
                        # This order needs this SKU
                        if order_id not in order_pickups:
                            order_pickups[order_id] = step_idx
        
        # Check for issues
        for order_id, delivery_step in order_deliveries.items():
            pickup_step = order_pickups.get(order_id, None)
            
            if pickup_step is None:
                issue = f"Route {route_idx+1}: Order {order_id} delivered at step {delivery_step} but no pickup found!"
                all_issues.append(issue)
                print(f"[X] {issue}")
            elif pickup_step >= delivery_step:
                issue = f"Route {route_idx+1}: Order {order_id} pickup at step {pickup_step} AFTER delivery at step {delivery_step}"
                all_issues.append(issue)
                print(f"[X] {issue}")
    
    if not all_issues:
        print("[OK] All pickups happen before deliveries")
    
    return all_issues

def check_path_continuity(env, solution):
    """Check if paths between steps are valid"""
    
    print("\n" + "=" * 80)
    print("PATH CONTINUITY CHECK")
    print("=" * 80)
    
    road_data = env.get_road_network_data()
    adjacency = road_data.get('adjacency_list', {})
    
    routes = solution.get('routes', [])
    all_issues = []
    
    for route_idx, route in enumerate(routes[:3]):  # Check first 3 routes
        vehicle_id = route['vehicle_id']
        steps = route['steps']
        
        print(f"\nRoute {route_idx+1} ({vehicle_id}):")
        
        for i in range(len(steps) - 1):
            current_node = steps[i]['node_id']
            next_node = steps[i+1]['node_id']
            
            # Check if there's a direct edge
            if current_node in adjacency:
                neighbors = adjacency[current_node]
                if next_node not in neighbors and current_node != next_node:
                    issue = f"  Step {i} -> {i+1}: No direct edge from {current_node} to {next_node}"
                    all_issues.append(issue)
                    print(f"[X] {issue}")
    
    if not all_issues:
        print("[OK] All consecutive steps have valid edges")
    
    return all_issues

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("DEEP ROUTE VALIDATION DEBUGGER")
    print("=" * 80)
    
    # Test on default scenario
    env = LogisticsEnvironment()
    
    print("\nGenerating solution with Solver 69...")
    solution = solver(env)
    
    print(f"Solution has {len(solution.get('routes', []))} routes")
    
    # Run all checks
    inspect_route_details(env, solution)
    
    pickup_issues = compare_pickup_delivery_order(env, solution)
    
    path_issues = check_path_continuity(env, solution)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL DIAGNOSIS")
    print("=" * 80)
    
    total_issues = len(pickup_issues) + len(path_issues)
    
    if total_issues == 0:
        print("[OK] No critical issues found in route structure!")
        print("   The validation failure may be due to other factors.")
    else:
        print(f"[X] Found {total_issues} critical issues:")
        print(f"   - Pickup/Delivery ordering: {len(pickup_issues)} issues")
        print(f"   - Path continuity: {len(path_issues)} issues")
        
        print("\nMOST CRITICAL FIXES NEEDED:")
        if pickup_issues:
            print("   1. Ensure ALL pickups happen BEFORE deliveries")
            print("   2. Group all pickups at warehouse visits first")
        if path_issues:
            print("   3. Add intermediate nodes between non-adjacent steps")
            print("   4. Use proper pathfinding (Dijkstra) for all segments")
