"""
Test solver 77 with DQN debugging enabled
"""

from Ne3Na3_solver_84 import solver
from robin_logistics import LogisticsEnvironment
import time

print("=" * 80)
print("SOLVER 77 - DEBUG TEST")
print("=" * 80)
print("\nNote: This will test on the default scenario with DQN debugging enabled\n")

start_time = time.time()
env = LogisticsEnvironment()

# Run with debugging enabled
result = solver(env, debug=True)
solve_time = time.time() - start_time

print("\n" + "="*80)
print("VALIDATION & RESULTS")
print("="*80)

# Validate
is_valid, msg, details = env.validate_solution_complete(result)

print(f"\nValid: {is_valid}")
if not is_valid:
    print(f"Validation error: {msg}")
    if details:
        print(f"Details: {details}")

# Get fulfillment
fulfillment = env.get_solution_fulfillment_summary(result)
fulfilled = fulfillment.get('fully_fulfilled_orders', 0)
total = fulfillment['total_orders']
distance = fulfillment.get('total_distance', 0)

# Get cost
cost = env.calculate_solution_cost(result) if is_valid else None

print(f"Fulfillment: {fulfilled} / {total} ({100*fulfilled/total:.1f}%)")
print(f"Distance: {distance:.2f} km")
if cost:
    print(f"Cost: ${cost:,.2f}")
print(f"Solve Time: {solve_time:.2f}s")

print("\n" + "="*80)
