"""
Test solver 77 with debugging enabled
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
    
    # Validate
    is_valid, msg, details = env.validate_solution_complete(result)
    
    # Get fulfillment
    fulfillment = env.get_solution_fulfillment_summary(result)
    fulfilled = fulfillment.get('fully_fulfilled_orders', 0)
    total = fulfillment['total_orders']
    distance = fulfillment.get('total_distance', 0)
    
    # Get cost
    cost = env.calculate_solution_cost(result) if is_valid else None
    
    result_data = {
        'seed': seed,
        'valid': is_valid,
        'fulfilled': fulfilled,
        'total': total,
        'fulfillment_pct': (fulfilled / total * 100) if total > 0 else 0,
        'distance': distance,
        'cost': cost,
        'solve_time': solve_time
    }
    results.append(result_data)
    
    print(f"Valid: {is_valid}")
    if not is_valid:
        print(f"Error: {msg}")
    print(f"Fulfillment: {fulfilled}/{total} ({result_data['fulfillment_pct']:.1f}%)")
    print(f"Distance: {distance:.2f} km")
    print(f"Cost: ${cost:,.2f}" if cost else "Cost: N/A")
    print(f"Solve time: {solve_time:.2f}s")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Seed':<10} {'Valid':<8} {'Fulfilled':<15} {'Fulfillment %':<15} {'Distance':<15} {'Cost':<15}")
print("-" * 80)

for r in results:
    print(f"{r['seed']:<10} {str(r['valid']):<8} {r['fulfilled']}/{r['total']:<10} "
          f"{r['fulfillment_pct']:>6.1f}%        "
          f"{r['distance']:>10.2f} km   "
          f"${r['cost']:>10,.0f}" if r['cost'] else "N/A")

print("-" * 80)
avg_fulfillment = sum(r['fulfillment_pct'] for r in results) / len(results)
avg_distance = sum(r['distance'] for r in results) / len(results)
avg_cost = sum(r['cost'] for r in results if r['cost']) / sum(1 for r in results if r['cost'])
print(f"{'AVERAGE':<10} {'':<8} {'':<15} {avg_fulfillment:>6.1f}%        "
      f"{avg_distance:>10.2f} km   ${avg_cost:>10,.0f}")
