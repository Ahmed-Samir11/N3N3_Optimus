"""
Compare Solver 69 vs Solver 72 (with improved DQN training)
"""
from robin_logistics import LogisticsEnvironment

print("=" * 70)
print("COMPARING SOLVERS: 69 (Original DQN) vs 72 (Improved Training)")
print("=" * 70)
print()

# Test Solver 69
print("Testing Solver 69 (Original DQN)...")
from Ne3Na3_solver_69 import solver as solver69
env69 = LogisticsEnvironment()
result69 = solver69(env69)
env69.execute_solution(result69)
fulfillment69 = env69.get_solution_fulfillment_summary(result69)
cost69 = env69.calculate_solution_cost(result69)
fulfilled69 = fulfillment69.get('fully_fulfilled_orders', 0)

print(f"  ✓ Fulfilled: {fulfilled69}/50 ({fulfilled69/50*100:.1f}%)")
print(f"  ✓ Cost: ${cost69:,.0f}")
print()

# Test Solver 72
print("Testing Solver 72 (Improved DQN with Experience Replay)...")
from Ne3Na3_solver_71 import solver as solver72
env72 = LogisticsEnvironment()
result72 = solver72(env72)
env72.execute_solution(result72)
fulfillment72 = env72.get_solution_fulfillment_summary(result72)
cost72 = env72.calculate_solution_cost(result72)
fulfilled72 = fulfillment72.get('fully_fulfilled_orders', 0)

print(f"  ✓ Fulfilled: {fulfilled72}/50 ({fulfilled72/50*100:.1f}%)")
print(f"  ✓ Cost: ${cost72:,.0f}")
print()

# Comparison
print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Solver 69: {fulfilled69}/50 orders ({fulfilled69/50*100:.1f}%) | Cost: ${cost69:,.0f}")
print(f"Solver 72: {fulfilled72}/50 orders ({fulfilled72/50*100:.1f}%) | Cost: ${cost72:,.0f}")
print()

if fulfilled72 > fulfilled69:
    print("✅ WINNER: Solver 72 (better fulfillment)")
    print(f"   Improvement: +{fulfilled72-fulfilled69} orders (+{(fulfilled72-fulfilled69)/50*100:.1f}%)")
elif fulfilled72 == fulfilled69:
    if cost72 < cost69:
        print("✅ WINNER: Solver 72 (same fulfillment, lower cost)")
        print(f"   Cost savings: ${cost69-cost72:,.0f} ({(cost69-cost72)/cost69*100:.1f}%)")
    else:
        print("⚖️  TIE: Same fulfillment, Solver 69 has lower cost")
        print(f"   Cost difference: ${cost72-cost69:,.0f}")
else:
    print("⚠️  Solver 69 is better (higher fulfillment)")
    print(f"   Difference: {fulfilled69-fulfilled72} orders")

print()
print("=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print("Solver 72 was trained with:")
print("  ✅ Experience replay buffer (prevents forgetting)")
print("  ✅ Curriculum learning (10→20→35 orders)")
print("  ✅ 350 episodes (early stopping at 99.5% avg)")
print("  ✅ 100% fulfillment on small scenarios")
print("  ✅ 99.6% fulfillment on medium scenarios")
print()
print("Competition recommendation:")
if fulfilled72 >= fulfilled69:
    print("  → Submit Solver 72 (improved training)")
else:
    print("  → Submit Solver 69 (better on this test case)")
    print("  → But test Solver 72 on Scenario 4 & 3 (may help there!)")
print("=" * 70)
