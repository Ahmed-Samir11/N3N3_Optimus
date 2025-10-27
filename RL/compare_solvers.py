"""Compare Solver 69 vs 71"""
from robin_logistics import LogisticsEnvironment

print("=" * 60)
print("Testing Solver 69 (Original DQN)")
print("=" * 60)

from Ne3Na3_solver_69 import solver as solver69
env = LogisticsEnvironment()
result69 = solver69(env)
env.execute_solution(result69)
fulfillment69 = env.get_solution_fulfillment_summary(result69)
cost69 = env.calculate_solution_cost(result69)

print(f"Fulfilled: {fulfillment69.get('fully_fulfilled_orders', 0)}/{len(env.get_all_order_ids())}")
print(f"Cost: ${cost69:,.0f}")

print("\n" + "=" * 60)
print("Testing Solver 71 (DQN + Greedy Fallback)")
print("=" * 60)

from Ne3Na3_solver_70 import solver as solver71
env2 = LogisticsEnvironment()
result71 = solver71(env2)
env2.execute_solution(result71)
fulfillment71 = env2.get_solution_fulfillment_summary(result71)
cost71 = env2.calculate_solution_cost(result71)

print(f"Fulfilled: {fulfillment71.get('fully_fulfilled_orders', 0)}/{len(env2.get_all_order_ids())}")
print(f"Cost: ${cost71:,.0f}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Solver 69: {fulfillment69.get('fully_fulfilled_orders', 0)}/50 orders (${cost69:,.0f})")
print(f"Solver 71: {fulfillment71.get('fully_fulfilled_orders', 0)}/50 orders (${cost71:,.0f})")

if fulfillment71.get('fully_fulfilled_orders', 0) > fulfillment69.get('fully_fulfilled_orders', 0):
    print("\n✅ Solver 71 is BETTER (more fulfillment)")
elif fulfillment71.get('fully_fulfilled_orders', 0) == fulfillment69.get('fully_fulfilled_orders', 0):
    if cost71 < cost69:
        print("\n✅ Solver 71 is BETTER (same fulfillment, lower cost)")
    else:
        print("\n⚠️  Same fulfillment, Solver 69 has lower cost")
else:
    print("\n⚠️  Solver 69 is BETTER")
