"""Compare solver 84 vs solver 90 performance"""
from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_84 import solver as solver84
from Ne3Na3_solver_93 import solver as solver90

# Test solver 84
print("=== SOLVER 84 (Baseline) ===")
env84 = LogisticsEnvironment()
result84 = solver84(env84)
cost84 = env84.calculate_solution_cost(result84)
fulfillment84 = env84.get_solution_fulfillment_summary(result84)
vehicles84 = len([r for r in result84['routes'] if r['steps']])
print(f"Fulfillment: {fulfillment84.get('fully_fulfilled_orders', 0)}/50")
print(f"Cost: ${cost84:,.2f}")
print(f"Vehicles used: {vehicles84}")

# Test solver 90
print("\n=== SOLVER 90 (Optimized) ===")
env90 = LogisticsEnvironment()
result90 = solver90(env90)
cost90 = env90.calculate_solution_cost(result90)
fulfillment90 = env90.get_solution_fulfillment_summary(result90)
vehicles90 = len([r for r in result90['routes'] if r['steps']])
print(f"Fulfillment: {fulfillment90.get('fully_fulfilled_orders', 0)}/50")
print(f"Cost: ${cost90:,.2f}")
print(f"Vehicles used: {vehicles90}")

# Calculate improvements
print("\n=== IMPROVEMENTS ===")
cost_savings = cost84 - cost90
cost_pct = 100 * cost_savings / cost84 if cost84 > 0 else 0
vehicle_reduction = vehicles84 - vehicles90
print(f"Cost savings: ${cost_savings:,.2f} ({cost_pct:.1f}%)")
print(f"Vehicle reduction: {vehicle_reduction}")
print(f"Fulfillment maintained: {fulfillment90.get('fully_fulfilled_orders', 0) == 50}")
