"""Test solver 84 vs solver 91 (aggressive optimization)"""
from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_84 import solver as solver84
from Ne3Na3_solver_93 import solver as solver91
import time

seeds = [42, 100, 200, 300, 400, 500]

print("=" * 80)
print("AGGRESSIVE OPTIMIZATION TEST: Solver 84 vs Solver 91")
print("=" * 80)

results_84 = []
results_91 = []

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
    
    print(f"  Fulfillment: {fulfilled84}/{total_orders}")
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
    
    # Test solver 91
    print("\n[Solver 91 - AGGRESSIVE]")
    env91 = LogisticsEnvironment()
    env91.generate_new_scenario(seed=seed)
    start = time.time()
    result91 = solver91(env91)
    time91 = time.time() - start
    
    cost91 = env91.calculate_solution_cost(result91)
    fulfillment91 = env91.get_solution_fulfillment_summary(result91)
    fulfilled91 = fulfillment91.get('fully_fulfilled_orders', 0)
    vehicles91 = len([r for r in result91['routes'] if r['steps']])
    
    print(f"  Fulfillment: {fulfilled91}/{total_orders}")
    print(f"  Cost: ${cost91:,.2f}")
    print(f"  Vehicles: {vehicles91}")
    print(f"  Time: {time91:.2f}s")
    
    results_91.append({
        'seed': seed,
        'fulfillment': fulfilled91,
        'total': total_orders,
        'cost': cost91,
        'vehicles': vehicles91,
        'time': time91
    })
    
    # Show comparison
    cost_diff = cost84 - cost91
    cost_pct = 100 * cost_diff / cost84 if cost84 > 0 else 0
    vehicle_diff = vehicles84 - vehicles91
    fulfillment_maintained = (fulfilled91 == total_orders) if (fulfilled84 == total_orders) else (fulfilled91 == fulfilled84)
    
    print(f"\n[Impact]")
    print(f"  Cost savings: ${cost_diff:+,.2f} ({cost_pct:+.1f}%)")
    print(f"  Vehicle reduction: {vehicle_diff:+d}")
    print(f"  Fulfillment maintained: {fulfillment_maintained}")
    
    if cost_pct > 1:
        print(f"  💰 SIGNIFICANT SAVINGS!")
    elif cost_pct < -1:
        print(f"  ⚠️  Cost increased")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

avg_cost_84 = sum(r['cost'] for r in results_84) / len(results_84)
avg_cost_91 = sum(r['cost'] for r in results_91) / len(results_91)
avg_vehicles_84 = sum(r['vehicles'] for r in results_84) / len(results_84)
avg_vehicles_91 = sum(r['vehicles'] for r in results_91) / len(results_91)

cost_savings = avg_cost_84 - avg_cost_91
cost_pct_overall = 100 * cost_savings / avg_cost_84 if avg_cost_84 > 0 else 0
vehicle_reduction = avg_vehicles_84 - avg_vehicles_91

print(f"\nSolver 84 Average: ${avg_cost_84:,.2f}, {avg_vehicles_84:.1f} vehicles")
print(f"Solver 91 Average: ${avg_cost_91:,.2f}, {avg_vehicles_91:.1f} vehicles")
print(f"\nCost Savings: ${cost_savings:+,.2f} ({cost_pct_overall:+.1f}%)")
print(f"Vehicle Reduction: {vehicle_reduction:+.2f}")

cost_wins_91 = sum(1 for i in range(len(seeds)) if results_91[i]['cost'] < results_84[i]['cost'])
fulfillment_maintained_all = all(
    results_91[i]['fulfillment'] == results_91[i]['total']
    for i in range(len(seeds))
    if results_84[i]['fulfillment'] == results_84[i]['total']
)

print(f"\nWin Rate: {cost_wins_91}/{len(seeds)} scenarios")
print(f"Fulfillment: {'✅ Maintained' if fulfillment_maintained_all else '❌ Degraded'}")

if cost_pct_overall >= 3 and fulfillment_maintained_all:
    print(f"\n🎯 SUCCESS: {cost_pct_overall:.1f}% cost reduction achieved!")
elif cost_pct_overall >= 1:
    print(f"\n✅ GOOD: {cost_pct_overall:.1f}% cost reduction")
elif cost_pct_overall > 0:
    print(f"\n📊 MARGINAL: {cost_pct_overall:.1f}% cost reduction")
else:
    print(f"\n⚠️  NO IMPROVEMENT: {cost_pct_overall:.1f}% change")
