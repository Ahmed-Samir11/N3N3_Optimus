"""Test FIXED Solver 69 across all 6 scenarios"""
from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_69 import solver
import time

scenarios = ['default', '1', '2', '3', '4', '5']

print("="*80)
print("TESTING FIXED SOLVER 69 - ALL SCENARIOS")
print("="*80)
print()

results = []

for scenario_id in scenarios:
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_id}")
    print(f"{'='*80}")
    
    try:
        # Initialize environment
        if scenario_id == 'default':
            env = LogisticsEnvironment()
        else:
            env = LogisticsEnvironment(scenario_id=scenario_id)
        
        # Run solver
        start_time = time.time()
        solution = solver(env)
        solve_time = time.time() - start_time
        
        # Validate
        is_valid, msg, details = env.validate_solution_complete(solution)
        
        # Execute if valid
        if is_valid:
            success, exec_msg = env.execute_solution(solution)
            cost = env.calculate_solution_cost(solution)
            fulfillment = env.get_solution_fulfillment_summary(solution)
            
            fulfilled = fulfillment.get("fully_fulfilled_orders", 0)
            total = len(env.get_all_order_ids())
            fulfill_pct = 100 * fulfilled / total if total > 0 else 0
            
            print(f"✓ VALID: {details['valid_routes']}/{details['total_routes']} routes")
            print(f"✓ EXECUTED: {success}")
            print(f"✓ FULFILLMENT: {fulfilled}/{total} ({fulfill_pct:.1f}%)")
            print(f"✓ COST: ${cost:,.0f}")
            print(f"✓ SOLVE TIME: {solve_time:.1f}s")
            
            results.append({
                'scenario': scenario_id,
                'valid': True,
                'fulfillment': fulfill_pct,
                'cost': cost,
                'time': solve_time
            })
        else:
            print(f"✗ VALIDATION FAILED: {msg}")
            print(f"  Invalid routes: {details.get('invalid_routes', [])}")
            results.append({
                'scenario': scenario_id,
                'valid': False,
                'fulfillment': 0,
                'cost': 0,
                'time': solve_time
            })
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        results.append({
            'scenario': scenario_id,
            'valid': False,
            'fulfillment': 0,
            'cost': 0,
            'time': 0
        })

# Summary table
print("\n" + "="*80)
print("SUMMARY - FIXED SOLVER 69")
print("="*80)
print(f"{'Scenario':<12} {'Valid':<8} {'Fulfillment':<15} {'Cost':<12} {'Time':<8}")
print("-"*80)
for r in results:
    valid_str = "✓" if r['valid'] else "✗"
    fulfill_str = f"{r['fulfillment']:.1f}%" if r['valid'] else "N/A"
    cost_str = f"${r['cost']:,.0f}" if r['valid'] else "N/A"
    time_str = f"{r['time']:.1f}s" if r['time'] > 0 else "N/A"
    print(f"{r['scenario']:<12} {valid_str:<8} {fulfill_str:<15} {cost_str:<12} {time_str:<8}")

print("\n" + "="*80)
print("KEY IMPROVEMENTS TO CHECK:")
print("  - Scenario 3: Was 0% → Now ?")
print("  - Scenario 4: Was 11% → Now ?")
print("="*80)
