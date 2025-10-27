"""
Test solver 84 vs solver 90 across multiple random seeds
to ensure consistent performance improvements
"""
from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_84 import solver as solver84
from Ne3Na3_solver_93 import solver as solver90
import time

# Test with multiple seeds
seeds = [42, 100, 200, 300, 400, 500]

print("=" * 80)
print("MULTI-SEED COMPARISON: Solver 84 vs Solver 90")
print("=" * 80)

results_84 = []
results_90 = []

for seed in seeds:
    print(f"\n{'='*80}")
    print(f"SEED: {seed}")
    print(f"{'='*80}")
    
    # Test solver 84
    print("\n[Solver 84]")
    env84 = LogisticsEnvironment()
    env84.generate_new_scenario(seed=seed)
    start = time.time()
    result84 = solver84(env84)
    time84 = time.time() - start
    
    cost84 = env84.calculate_solution_cost(result84)
    fulfillment84 = env84.get_solution_fulfillment_summary(result84)
    fulfilled84 = fulfillment84.get('fully_fulfilled_orders', 0)
    total_orders = len(env84.get_all_order_ids())
    vehicles84 = len([r for r in result84['routes'] if r['steps']])
    
    print(f"  Fulfillment: {fulfilled84}/{total_orders} ({100*fulfilled84/total_orders:.1f}%)")
    print(f"  Cost: ${cost84:,.2f}")
    print(f"  Vehicles: {vehicles84}")
    print(f"  Time: {time84:.2f}s")
    
    results_84.append({
        'seed': seed,
        'fulfillment': fulfilled84,
        'total': total_orders,
        'cost': cost84,
        'vehicles': vehicles84,
        'time': time84
    })
    
    # Test solver 90
    print("\n[Solver 90]")
    env90 = LogisticsEnvironment()
    env90.generate_new_scenario(seed=seed)
    start = time.time()
    result90 = solver90(env90)
    time90 = time.time() - start
    
    cost90 = env90.calculate_solution_cost(result90)
    fulfillment90 = env90.get_solution_fulfillment_summary(result90)
    fulfilled90 = fulfillment90.get('fully_fulfilled_orders', 0)
    vehicles90 = len([r for r in result90['routes'] if r['steps']])
    
    print(f"  Fulfillment: {fulfilled90}/{total_orders} ({100*fulfilled90/total_orders:.1f}%)")
    print(f"  Cost: ${cost90:,.2f}")
    print(f"  Vehicles: {vehicles90}")
    print(f"  Time: {time90:.2f}s")
    
    results_90.append({
        'seed': seed,
        'fulfillment': fulfilled90,
        'total': total_orders,
        'cost': cost90,
        'vehicles': vehicles90,
        'time': time90
    })
    
    # Show comparison
    cost_diff = cost84 - cost90
    cost_pct = 100 * cost_diff / cost84 if cost84 > 0 else 0
    vehicle_diff = vehicles84 - vehicles90
    time_diff = time84 - time90
    
    print(f"\n[Comparison]")
    print(f"  Cost difference: ${cost_diff:+,.2f} ({cost_pct:+.1f}%)")
    print(f"  Vehicle difference: {vehicle_diff:+d}")
    print(f"  Time difference: {time_diff:+.2f}s")
    print(f"  Fulfillment maintained: {fulfilled90 == total_orders}")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")

avg_cost_84 = sum(r['cost'] for r in results_84) / len(results_84)
avg_cost_90 = sum(r['cost'] for r in results_90) / len(results_90)
avg_vehicles_84 = sum(r['vehicles'] for r in results_84) / len(results_84)
avg_vehicles_90 = sum(r['vehicles'] for r in results_90) / len(results_90)
avg_time_84 = sum(r['time'] for r in results_84) / len(results_84)
avg_time_90 = sum(r['time'] for r in results_90) / len(results_90)

total_fulfilled_84 = sum(r['fulfillment'] for r in results_84)
total_orders_84 = sum(r['total'] for r in results_84)
total_fulfilled_90 = sum(r['fulfillment'] for r in results_90)
total_orders_90 = sum(r['total'] for r in results_90)

print(f"\nSolver 84 (Baseline):")
print(f"  Average cost: ${avg_cost_84:,.2f}")
print(f"  Average vehicles: {avg_vehicles_84:.1f}")
print(f"  Average time: {avg_time_84:.2f}s")
print(f"  Overall fulfillment: {total_fulfilled_84}/{total_orders_84} ({100*total_fulfilled_84/total_orders_84:.1f}%)")

print(f"\nSolver 90 (Optimized):")
print(f"  Average cost: ${avg_cost_90:,.2f}")
print(f"  Average vehicles: {avg_vehicles_90:.1f}")
print(f"  Average time: {avg_time_90:.2f}s")
print(f"  Overall fulfillment: {total_fulfilled_90}/{total_orders_90} ({100*total_fulfilled_90/total_orders_90:.1f}%)")

cost_savings = avg_cost_84 - avg_cost_90
cost_pct_overall = 100 * cost_savings / avg_cost_84 if avg_cost_84 > 0 else 0
vehicle_reduction = avg_vehicles_84 - avg_vehicles_90
time_savings = avg_time_84 - avg_time_90

print(f"\nOverall Improvement:")
print(f"  Average cost savings: ${cost_savings:+,.2f} ({cost_pct_overall:+.1f}%)")
print(f"  Average vehicle reduction: {vehicle_reduction:+.2f}")
print(f"  Average time difference: {time_savings:+.2f}s")

# Count wins
cost_wins_90 = sum(1 for i in range(len(seeds)) if results_90[i]['cost'] < results_84[i]['cost'])
cost_wins_84 = sum(1 for i in range(len(seeds)) if results_84[i]['cost'] < results_90[i]['cost'])
ties = len(seeds) - cost_wins_90 - cost_wins_84

print(f"\nHead-to-Head Cost Performance:")
print(f"  Solver 90 wins: {cost_wins_90}/{len(seeds)}")
print(f"  Solver 84 wins: {cost_wins_84}/{len(seeds)}")
print(f"  Ties: {ties}/{len(seeds)}")

print(f"\n{'='*80}")
if cost_savings > 0 and total_fulfilled_90 == total_orders_90:
    print("✅ RESULT: Solver 90 shows consistent improvements!")
    print(f"   - Maintains 100% fulfillment")
    print(f"   - Reduces average cost by {cost_pct_overall:.1f}%")
    print(f"   - Adds timeout protection for reliability")
elif total_fulfilled_90 == total_orders_90:
    print("⚠️  RESULT: Performance is similar, but solver 90 adds timeout protection")
else:
    print("❌ RESULT: Solver 90 needs further optimization")
print(f"{'='*80}")
