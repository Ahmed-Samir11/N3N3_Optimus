from robin_logistics import LogisticsEnvironment
from Ne3Na3_solver_69 import solver
import json

env = LogisticsEnvironment()
result = solver(env)
validation = env.validate_solution_complete(result)
details = validation[2]

print("Invalid routes and errors:")
for r in details.get('invalid_routes', []):
    print(f"\nVehicle: {r['vehicle_id']}")
    print(json.dumps(r, indent=2))
