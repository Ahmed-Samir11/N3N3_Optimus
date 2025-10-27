"""Test Fixed Solver 69 - Multi-warehouse pickups"""
from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_69 import solver

print("Testing FIXED Solver 69 with multi-warehouse pickup logic...")
env = LogisticsEnvironment()
result = solver(env)

validation = env.validate_solution_complete(result)
is_valid = validation[0]
msg = validation[1] if len(validation) > 1 else "No message"
details = validation[2] if len(validation) > 2 else {}

print(f"\nValid: {is_valid}")
print(f"Message: {msg}")
print(f"\nValidation details:")
print(f"  Total routes: {details.get('total_routes', 0)}")
print(f"  Valid: {details.get('valid_count', 0)}")
print(f"  Invalid: {details.get('invalid_count', 0)}")

if 'invalid_routes' in details:
    print(f"\nInvalid routes:")
    for inv_route in details['invalid_routes'][:3]:
        print(f"  - Vehicle: {inv_route.get('vehicle_id', 'Unknown')}")
        errors = inv_route.get('errors', [])
        for err in errors[:5]:
            print(f"    * {err}")

if is_valid:
    success, exec_msg = env.execute_solution(result)
    print(f"\nExecuted: {success}")
    
    fulfillment = env.get_solution_fulfillment_summary(result)
    fulfilled = fulfillment.get("fully_fulfilled_orders", 0)
    print(f"Fulfilled: {fulfilled}/50 orders ({100*fulfilled/50:.0f}%)")
    
    cost = env.calculate_solution_cost(result)
    print(f"Cost: ${cost:,.0f}")
