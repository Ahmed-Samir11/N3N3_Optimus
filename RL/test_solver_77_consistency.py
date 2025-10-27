"""
Test solver 77 multiple times (no debug output)
"""

from Ne3Na3_solver_84 import solver
from robin_logistics import LogisticsEnvironment
import time

print("=" * 80)
print("SOLVER 77 - MULTIPLE RUNS TEST")
print("=" * 80)
print("\nNote: Running on the same scenario multiple times to check consistency\n")

num_runs = 5
results = []

for run in range(1, num_runs + 1):
    print(f"Run {run}/{num_runs}...", end=" ")
    
    start_time = time.time()
    env = LogisticsEnvironment()
    result = solver(env, debug=False)
    solve_time = time.time() - start_time
    
    # Validate
    is_valid, msg, details = env.validate_solution_complete(result)
    
    # Get fulfillment
    fulfillment = env.get_solution_fulfillment_summary(result)
    fulfilled = fulfillment.get('fully_fulfilled_orders', 0)
    total = fulfillment['total_orders']
    distance = fulfillment.get('total_distance', 0)
    
    # Get cost
    cost = env.calculate_solution_cost(result) if is_valid else None
    
    results.append({
        'run': run,
        'valid': is_valid,
        'fulfilled': fulfilled,
        'total': total,
        'fulfillment_pct': (fulfilled / total * 100) if total > 0 else 0,
        'distance': distance,
        'cost': cost,
        'solve_time': solve_time
    })
    
    cost_str = f"${cost:,.0f}" if cost else "N/A"
    print(f"✓ {fulfilled}/{total} orders ({100*fulfilled/total:.0f}%), {cost_str}, {solve_time:.1f}s")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Run':<8} {'Valid':<8} {'Fulfilled':<15} {'Fulfillment %':<15} {'Distance':<15} {'Cost':<15} {'Time':<10}")
print("-" * 80)

for r in results:
    print(f"{r['run']:<8} {str(r['valid']):<8} {r['fulfilled']}/{r['total']:<10} "
          f"{r['fulfillment_pct']:>6.1f}%        "
          f"{r['distance']:>10.2f} km   "
          f"${r['cost']:>10,.0f}   " if r['cost'] else "N/A            "
          f"{r['solve_time']:>6.1f}s")

print("-" * 80)
avg_fulfillment = sum(r['fulfillment_pct'] for r in results) / len(results)
avg_distance = sum(r['distance'] for r in results) / len(results)
avg_cost = sum(r['cost'] for r in results if r['cost']) / sum(1 for r in results if r['cost'])
avg_time = sum(r['solve_time'] for r in results) / len(results)
print(f"{'AVERAGE':<8} {'':<8} {'':<15} {avg_fulfillment:>6.1f}%        "
      f"{avg_distance:>10.2f} km   ${avg_cost:>10,.0f}   {avg_time:>6.1f}s")

print("\n" + "=" * 80)
