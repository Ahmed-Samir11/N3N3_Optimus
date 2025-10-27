"""Test solver 76"""

from Ne3Na3_solver_76 import solver
from robin_logistics import LogisticsEnvironment

env = LogisticsEnvironment()
print("Running solver 76...")
result = solver(env)

print(f"\nRoutes generated: {len(result.get('routes', []))}")

validation = env.validate_solution_complete(result)
print(f"Validation: {'PASSED' if validation[0] else 'FAILED'}")
print(f"Message: {validation[1]}")

if validation[2]:
    print("\nValidation details:")
    for key, value in validation[2].items():
        if isinstance(value, list) and value:
            print(f"  {key}: {len(value)} items")
            for item in value[:3]:
                print(f"    - {item}")
        elif value:
            print(f"  {key}: {value}")

fulfillment = env.get_solution_fulfillment_summary(result)
print(f"\nFulfillment: {fulfillment.get('fully_fulfilled_orders', 0)}/{fulfillment['total_orders']}")

if validation[0]:
    cost = env.calculate_solution_cost(result)
    print(f"Cost: ${cost:,.2f}")
