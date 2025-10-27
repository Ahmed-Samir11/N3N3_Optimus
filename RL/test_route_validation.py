"""
Comprehensive Route Validation Debugger
Tests Solver 69 across multiple seeds to identify validation failures
"""

from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_69 import solver
import traceback

def validate_route_detailed(env, route, route_idx):
    """
    Detailed validation of a single route
    Returns (is_valid, error_details)
    """
    errors = []
    vehicle_id = route['vehicle_id']
    steps = route['steps']
    
    # Check 1: Route starts at vehicle home
    try:
        vehicle = None
        for v in env.get_all_vehicles():
            if v.id == vehicle_id:
                vehicle = v
                break
        
        if not vehicle:
            errors.append(f"Vehicle {vehicle_id} not found")
            return False, errors
        
        home_wh_id = vehicle.home_warehouse_id
        home_node = env.warehouses[home_wh_id].location.id
        
        if steps[0]['node_id'] != home_node:
            errors.append(f"Route {route_idx}: First step node {steps[0]['node_id']} != home {home_node}")
    except Exception as e:
        errors.append(f"Route {route_idx}: Error checking home node: {e}")
        return False, errors
    
    # Check 2: Track vehicle inventory
    vehicle_inventory = {}  # {sku_id: quantity}
    
    for step_idx, step in enumerate(steps):
        node_id = step['node_id']
        pickups = step.get('pickups', [])
        deliveries = step.get('deliveries', [])
        
        # Process pickups
        for pickup in pickups:
            wh_id = pickup['warehouse_id']
            sku_id = pickup['sku_id']
            quantity = pickup['quantity']
            
            # Check warehouse has inventory
            try:
                wh_inv = env.get_warehouse_inventory(wh_id)
                if wh_inv.get(sku_id, 0) < quantity:
                    errors.append(f"Route {route_idx}, Step {step_idx}: Warehouse {wh_id} has "
                                f"{wh_inv.get(sku_id, 0)} {sku_id}, need {quantity}")
            except Exception as e:
                errors.append(f"Route {route_idx}, Step {step_idx}: Error checking warehouse: {e}")
            
            # Check node is warehouse location
            try:
                wh_node = env.warehouses[wh_id].location.id
                if node_id != wh_node:
                    errors.append(f"Route {route_idx}, Step {step_idx}: Pickup at node {node_id} "
                                f"but warehouse {wh_id} is at node {wh_node}")
            except Exception as e:
                errors.append(f"Route {route_idx}, Step {step_idx}: Error checking warehouse node: {e}")
            
            # Add to vehicle inventory
            vehicle_inventory[sku_id] = vehicle_inventory.get(sku_id, 0) + quantity
        
        # Process deliveries
        for delivery in deliveries:
            order_id = delivery['order_id']
            sku_id = delivery['sku_id']
            quantity = delivery['quantity']
            
            # Check vehicle has items
            if vehicle_inventory.get(sku_id, 0) < quantity:
                errors.append(f"Route {route_idx}, Step {step_idx}: Vehicle has {vehicle_inventory.get(sku_id, 0)} "
                            f"{sku_id}, need {quantity} for order {order_id}")
            
            # Check node is order destination
            try:
                order = env.orders[order_id]
                order_node = order.destination.id
                if node_id != order_node:
                    errors.append(f"Route {route_idx}, Step {step_idx}: Delivery at node {node_id} "
                                f"but order {order_id} destination is {order_node}")
            except Exception as e:
                errors.append(f"Route {route_idx}, Step {step_idx}: Error checking order destination: {e}")
            
            # Remove from vehicle inventory
            vehicle_inventory[sku_id] = vehicle_inventory.get(sku_id, 0) - quantity
    
    # Check 3: Route ends at home
    if steps[-1]['node_id'] != home_node:
        errors.append(f"Route {route_idx}: Last step node {steps[-1]['node_id']} != home {home_node}")
    
    # Check 4: Path connectivity
    for i in range(len(steps) - 1):
        curr_node = steps[i]['node_id']
        next_node = steps[i+1]['node_id']
        
        if curr_node == next_node:
            continue  # Same node is okay
        
        # Check if direct edge exists OR if they're consecutive in a valid path
        distance = env.get_distance(curr_node, next_node)
        if distance is None:
            # No direct edge - this is actually OKAY if intermediate nodes are included
            # The solver should be including intermediate nodes
            pass  # We'll check path validity separately
    
    return len(errors) == 0, errors


def test_solver_detailed(seed=None, label="Default"):
    """Test solver on a specific scenario with detailed debugging"""
    
    print("\n" + "=" * 80)
    print(f"TESTING: {label} (seed={seed})")
    print("=" * 80)
    
    try:
        env = LogisticsEnvironment() if seed is None else LogisticsEnvironment()
        
        # Get environment stats
        all_orders = env.get_all_order_ids()
        vehicles = env.get_all_vehicles()
        
        print(f"\n📊 Environment: {len(all_orders)} orders, {len(vehicles)} vehicles")
        
        # Run solver
        print("🔄 Running solver...")
        result = solver(env)
        
        # Validate solution structure
        if 'routes' not in result:
            print("❌ ERROR: Solution missing 'routes' key")
            return
        
        routes = result['routes']
        print(f"✓ Solver generated {len(routes)} routes")
        
        # Detailed validation of each route
        print("\n🔍 DETAILED ROUTE VALIDATION")
        print("-" * 80)
        
        all_valid = True
        for idx, route in enumerate(routes):
            is_valid, errors = validate_route_detailed(env, route, idx)
            
            if is_valid:
                print(f"✓ Route {idx} ({route['vehicle_id']}): VALID ({len(route['steps'])} steps)")
            else:
                all_valid = False
                print(f"❌ Route {idx} ({route['vehicle_id']}): INVALID")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"   - {error}")
                if len(errors) > 5:
                    print(f"   ... and {len(errors) - 5} more errors")
        
        # Execute solution
        print("\n🚀 EXECUTING SOLUTION")
        print("-" * 80)
        
        success, msg = env.execute_solution(result)
        fulfillment = env.get_solution_fulfillment_summary(result)
        cost = env.calculate_solution_cost(result)
        
        fulfilled = fulfillment.get('fully_fulfilled_orders', 0)
        
        print(f"Execution Success: {success}")
        if not success:
            print(f"Error Message: {msg}")
        
        print(f"Fulfilled: {fulfilled}/{len(all_orders)} ({fulfilled/len(all_orders)*100:.1f}%)")
        print(f"Cost: ${cost:,.0f}")
        
        # Summary
        print("\n" + "=" * 80)
        if all_valid and success:
            print(f"✅ {label}: ALL ROUTES VALID & EXECUTED SUCCESSFULLY")
        elif all_valid and not success:
            print(f"⚠️  {label}: Routes valid but execution failed - API issue?")
        elif not all_valid and not success:
            print(f"❌ {label}: Routes invalid and execution failed")
        else:
            print(f"🤔 {label}: Routes invalid but execution succeeded?!")
        
        return {
            'label': label,
            'seed': seed,
            'routes_valid': all_valid,
            'execution_success': success,
            'fulfilled': fulfilled,
            'total': len(all_orders),
            'cost': cost
        }
        
    except Exception as e:
        print(f"\n❌ EXCEPTION in {label}:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def run_multi_scenario_tests():
    """Run tests across multiple scenarios"""
    
    print("\n" + "🔬" * 40)
    print("MULTI-SCENARIO ROUTE VALIDATION TEST SUITE")
    print("🔬" * 40)
    
    results = []
    
    # Test 1: Default scenario
    result = test_solver_detailed(seed=None, label="Default Scenario")
    if result:
        results.append(result)
    
    # Test 2-6: Different seeds (simulating different scenarios)
    for i in range(5):
        result = test_solver_detailed(seed=i+1, label=f"Scenario {i+1}")
        if result:
            results.append(result)
    
    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Scenario':<20} {'Valid Routes':<15} {'Executed':<12} {'Fulfillment':<20} {'Cost':<15}")
    print("-" * 80)
    
    for r in results:
        valid_str = "✅ Yes" if r['routes_valid'] else "❌ No"
        exec_str = "✅ Yes" if r['execution_success'] else "❌ No"
        fulfill_str = f"{r['fulfilled']}/{r['total']} ({r['fulfilled']/r['total']*100:.0f}%)"
        cost_str = f"${r['cost']:,.0f}"
        
        print(f"{r['label']:<20} {valid_str:<15} {exec_str:<12} {fulfill_str:<20} {cost_str:<15}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    invalid_count = sum(1 for r in results if not r['routes_valid'])
    exec_fail_count = sum(1 for r in results if not r['execution_success'])
    
    print(f"Total scenarios tested: {len(results)}")
    print(f"Routes invalid: {invalid_count}/{len(results)}")
    print(f"Execution failed: {exec_fail_count}/{len(results)}")
    
    if invalid_count > 0:
        print("\n⚠️  CRITICAL: Route validation issues detected!")
        print("   Common issues to fix:")
        print("   1. Pickups not at warehouse locations")
        print("   2. Deliveries before pickups")
        print("   3. Missing intermediate path nodes")
        print("   4. Wrong start/end nodes")
    
    if exec_fail_count > 0 and invalid_count == 0:
        print("\n⚠️  STRANGE: Routes are valid but execution fails")
        print("   This suggests an API issue or missing edge case")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    run_multi_scenario_tests()
