"""Test Solver 75 - Hybrid DQN + Memetic"""

from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_76 import solver
import time

print("=" * 80)
print("TESTING SOLVER 75: HYBRID DQN + MEMETIC")
print("=" * 80)
print()

# Initialize environment
env = LogisticsEnvironment()

# Run solver
print("Running Solver 75...")
print("  Phase 1: DQN order-warehouse assignment")
print("  Phase 2: Initial route building")
print("  Phase 3: Memetic optimization (15s budget)")
print()

start_time = time.time()
solution = solver(env)
solve_time = time.time() - start_time

print(f"✓ Solver completed in {solve_time:.1f}s")
print(f"  Routes generated: {len(solution['routes'])}")
print()

# Validate
print("Validating solution...")
is_valid, msg, details = env.validate_solution_complete(solution)

if is_valid:
    print(f"✅ VALIDATION: PASSED ({details['valid_routes']}/{details['total_routes']} routes)")
    print()
    
    # Execute
    success, exec_msg = env.execute_solution(solution)
    cost = env.calculate_solution_cost(solution)
    fulfillment = env.get_solution_fulfillment_summary(solution)
    
    fulfilled = fulfillment.get("fully_fulfilled_orders", 0)
    total = len(env.get_all_order_ids())
    fulfill_pct = 100 * fulfilled / total
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Fulfillment:  {fulfill_pct:.1f}% ({fulfilled}/{total} orders)")
    print(f"Cost:         ${cost:,.0f}")
    print(f"Solve Time:   {solve_time:.1f}s")
    print()
    
    # Compare with Solver 69 baseline
    print("=" * 80)
    print("COMPARISON WITH SOLVER 69 (BASELINE)")
    print("=" * 80)
    print("Solver 69 (DQN only):")
    print("  - Fulfillment: 96% (48/50)")
    print("  - Cost: $4,052")
    print()
    print(f"Solver 75 (DQN + Memetic):")
    print(f"  - Fulfillment: {fulfill_pct:.1f}% ({fulfilled}/{total})")
    print(f"  - Cost: ${cost:,.0f}")
    print()
    
    if fulfill_pct >= 96:
        print("✅ Fulfillment maintained or improved!")
    else:
        print("⚠️  Fulfillment decreased - needs tuning")
    
    if cost <= 4052:
        print("✅ Cost improved!")
    elif cost <= 5000:
        print("⚠️  Cost slightly higher but acceptable")
    else:
        print("❌ Cost significantly increased")
        
else:
    print(f"❌ VALIDATION FAILED: {msg}")
    print(f"   Valid routes: {details.get('valid_routes', 0)}/{details.get('total_routes', 0)}")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
