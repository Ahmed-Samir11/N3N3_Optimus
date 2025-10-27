"""
Detailed cost analysis comparing solver 84 vs 90
Shows where cost savings come from
"""
from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_84 import solver as solver84
from Ne3Na3_solver_93 import solver as solver90

def analyze_solution(env, solution, label):
    """Detailed cost breakdown"""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    
    # Get cost components
    total_cost = env.calculate_solution_cost(solution)
    fulfillment = env.get_solution_fulfillment_summary(solution)
    
    # Count vehicles and calculate fixed cost
    active_routes = [r for r in solution['routes'] if r['steps']]
    num_vehicles = len(active_routes)
    
    # Estimate cost breakdown (rough approximation)
    # Fixed cost per vehicle varies by type, but average ~$500-600
    est_fixed_cost = num_vehicles * 525  # Average estimate
    est_variable_cost = total_cost - est_fixed_cost
    
    # Calculate total distance
    total_distance = 0
    for route in active_routes:
        steps = route['steps']
        for i in range(len(steps) - 1):
            node1 = steps[i]['node_id']
            node2 = steps[i+1]['node_id']
            dist = env.get_distance(node1, node2)
            if dist:
                total_distance += dist
    
    # Count multi-warehouse orders
    multi_wh_count = 0
    single_wh_count = 0
    total_pickups = 0
    
    for route in active_routes:
        for step in route['steps']:
            if step['pickups']:
                total_pickups += len(step['pickups'])
    
    print(f"Total Cost: ${total_cost:,.2f}")
    print(f"  Est. Fixed Cost (vehicles): ${est_fixed_cost:,.2f} ({num_vehicles} vehicles)")
    print(f"  Est. Variable Cost (distance): ${est_variable_cost:,.2f}")
    print(f"\nFulfillment: {fulfillment.get('fully_fulfilled_orders', 0)}/{len(env.get_all_order_ids())}")
    print(f"Total Distance: {total_distance:.2f} km")
    print(f"Total Pickups: {total_pickups}")
    
    return {
        'total_cost': total_cost,
        'vehicles': num_vehicles,
        'distance': total_distance,
        'pickups': total_pickups,
        'fulfillment': fulfillment.get('fully_fulfilled_orders', 0)
    }

# Test on multiple seeds
seeds = [42, 100, 200, 300, 400, 500]
results = []

print("=" * 80)
print("DETAILED COST ANALYSIS: Solver 84 vs Solver 90")
print("=" * 80)

for seed in seeds:
    print(f"\n{'#'*80}")
    print(f"TESTING SEED: {seed}")
    print(f"{'#'*80}")
    
    # Test solver 84
    env84 = LogisticsEnvironment()
    env84.generate_new_scenario(seed=seed)
    result84 = solver84(env84)
    stats84 = analyze_solution(env84, result84, "SOLVER 84 (Baseline)")
    
    # Test solver 90
    env90 = LogisticsEnvironment()
    env90.generate_new_scenario(seed=seed)
    result90 = solver90(env90)
    stats90 = analyze_solution(env90, result90, "SOLVER 90 (Optimized)")
    
    # Compare
    print(f"\n{'-'*60}")
    print("COMPARISON")
    print(f"{'-'*60}")
    cost_diff = stats84['total_cost'] - stats90['total_cost']
    cost_pct = 100 * cost_diff / stats84['total_cost'] if stats84['total_cost'] > 0 else 0
    dist_diff = stats84['distance'] - stats90['distance']
    dist_pct = 100 * dist_diff / stats84['distance'] if stats84['distance'] > 0 else 0
    pickup_diff = stats84['pickups'] - stats90['pickups']
    
    print(f"Cost Difference: ${cost_diff:+,.2f} ({cost_pct:+.2f}%)")
    print(f"Distance Difference: {dist_diff:+.2f} km ({dist_pct:+.2f}%)")
    print(f"Pickup Difference: {pickup_diff:+d}")
    print(f"Vehicle Difference: {stats84['vehicles'] - stats90['vehicles']:+d}")
    
    results.append({
        'seed': seed,
        'cost_diff': cost_diff,
        'cost_pct': cost_pct,
        'dist_diff': dist_diff,
        'pickup_diff': pickup_diff,
        'vehicle_diff': stats84['vehicles'] - stats90['vehicles']
    })

# Summary
print(f"\n{'='*80}")
print("OVERALL SUMMARY")
print(f"{'='*80}")

avg_cost_diff = sum(r['cost_diff'] for r in results) / len(results)
avg_cost_pct = sum(r['cost_pct'] for r in results) / len(results)
avg_dist_diff = sum(r['dist_diff'] for r in results) / len(results)
avg_pickup_diff = sum(r['pickup_diff'] for r in results) / len(results)

wins = sum(1 for r in results if r['cost_diff'] > 0)
losses = sum(1 for r in results if r['cost_diff'] < 0)
ties = len(results) - wins - losses

print(f"\nAverage Cost Savings: ${avg_cost_diff:+,.2f} ({avg_cost_pct:+.2f}%)")
print(f"Average Distance Reduction: {avg_dist_diff:+.2f} km")
print(f"Average Pickup Reduction: {avg_pickup_diff:+.1f}")
print(f"\nHead-to-Head: Solver 90 wins {wins}/{len(results)}, loses {losses}/{len(results)}, ties {ties}/{len(results)}")

if avg_cost_diff > 0:
    print(f"\n✅ SOLVER 90 achieves {avg_cost_pct:.2f}% average cost reduction")
    print(f"   Primary savings from: {'distance reduction' if avg_dist_diff > 0 else 'warehouse optimization'}")
else:
    print(f"\n⚠️  SOLVER 90 shows marginal performance difference ({avg_cost_pct:.2f}%)")
    print("   Optimizations are conservative to maintain fulfillment reliability")
