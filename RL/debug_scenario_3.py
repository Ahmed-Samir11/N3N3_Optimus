"""Debug scenario 3 - why 0% fulfillment?"""
from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_84 import solver
import time

print("=" * 80)
print("DEBUGGING SCENARIO 3 - 0% FULFILLMENT")
print("=" * 80)

env = LogisticsEnvironment()
# Scenario 3 is the default environment
print(f"\nEnvironment info:")
print(f"Total orders: {len(env.get_all_order_ids())}")
print(f"Total vehicles: {len(env.get_all_vehicles())}")
print(f"Warehouses: {list(env.warehouses.keys())}")

try:
    start = time.time()
    result = solver(env)
    solve_time = time.time() - start
    
    print(f"\nSolver completed in {solve_time:.2f}s")
    print(f"Routes generated: {len(result.get('routes', []))}")
    
    # Validate
    validation = env.validate_solution_complete(result)
    if isinstance(validation, dict):
        print(f"\nValidation:")
        print(f"  Valid routes: {validation.get('valid_count', 0)}/{validation.get('total_routes', 0)}")
        invalid = validation.get('invalid_routes', [])
        if invalid:
            print(f"  Invalid routes: {len(invalid)}")
            for i, err in enumerate(invalid[:3]):
                print(f"    {i+1}. {err}")
    
    # Execute
    success, msg = env.execute_solution(result)
    print(f"\nExecution: {'✓ SUCCESS' if success else '✗ FAILED'}")
    if not success:
        print(f"  Error: {msg}")
    
    # Get fulfillment
    fulfillment = env.get_solution_fulfillment_summary(result)
    fulfilled = fulfillment.get('fully_fulfilled_orders', 0)
    total = len(env.get_all_order_ids())
    
    print(f"\nFulfillment: {fulfilled}/{total} ({100*fulfilled/total:.1f}%)")
    
    if success:
        cost = env.calculate_solution_cost(result)
        print(f"Cost: ${cost:,.2f}")
    
except Exception as e:
    print(f"\n✗ SOLVER CRASHED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
