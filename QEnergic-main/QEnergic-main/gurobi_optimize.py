# gurobi_optimize.py
"""
Solve QUBO with Gurobi. Usage:
    python gurobi_optimize.py --qubo_file QUBO.json --budget 10000 --costs_file costs.json
QUBO.json format: {"Q": [[...]]}
costs.json format: [cost_0, cost_1, ...] (cost for each candidate)
Prints: {"selected_indices": [...], "fval": ...}

This script performs QUBO optimization with a budget constraint:
- Objective: Minimize x'Qx (QUBO)
- Constraint: sum_i x_i * cost_i <= budget (select as many as possible within budget)
- x_i are binary variables (0 or 1)

This is a combinatorial optimization problem with a linear constraint (budget).
"""
import argparse
import json
import numpy as np
import time
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print(json.dumps({"error": "gurobipy not installed"}))
    exit(1)

def load_qubo(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['Q'])

def load_costs(costs_file):
    with open(costs_file, 'r') as f:
        return np.array(json.load(f))

def load_data(data_file):
    """Load data from the new data structure"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data

def solve_qubo_gurobi(Q, costs, budget):
    n = Q.shape[0]
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    # Objective: sum_i sum_j Q[i][j] x_i x_j
    obj = gp.quicksum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n))
    m.setObjective(obj, GRB.MINIMIZE)
    
    # The budget constraint is now fully encoded in the QUBO matrix.
    # The hard constraint below is removed to ensure we solve the same problem
    # as the quantum script (which uses soft constraints).
    # m.addConstr(gp.quicksum(x[i] * float(costs[i]) for i in range(n)) <= budget)
    
    m.optimize()
    if m.status == GRB.Status.INFEASIBLE:
        print(json.dumps({"debug": "INFEASIBLE", "budget": float(budget), "costs": costs.tolist()}))
        return None, None  # Infeasible
    xsol = [int(x[i].X > 0.5) for i in range(n)]
    selected = [i for i, v in enumerate(xsol) if v]
    total_cost = float(np.dot(xsol, costs))
    print(json.dumps({"debug": "SOLUTION", "budget": float(budget), "costs": costs.tolist(), "selected_indices": selected, "total_cost": total_cost}))
    return selected, m.objVal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubo_file', type=str, required=True)
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--costs_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, help='Optional data file for analysis')
    args = parser.parse_args()
    Q = load_qubo(args.qubo_file)
    costs = load_costs(args.costs_file)
    t0 = time.time()
    selected, fval = solve_qubo_gurobi(Q, costs, args.budget)
    elapsed = time.time() - t0
    if selected is None:
        print(json.dumps({"error": "No feasible solution under budget constraint", "selected_indices": [], "fval": None, "time_sec": elapsed}))
        exit(2)
    
    result = {"selected_indices": selected, "fval": fval, "time_sec": elapsed}
    
    # Add analysis if data file is provided
    if args.data_file:
        try:
            data = load_data(args.data_file)
            selected_data = [data[i] for i in selected]
            total_cost = sum(site["Installation_Cost_USD"] for site in selected_data)
            total_population = sum(site["Population_Coverage"] for site in selected_data)
            total_energy = sum(site["Energy_Capacity_kWh_day"] for site in selected_data)
            result.update({
                "total_cost": total_cost,
                "total_population": total_population,
                "total_energy": total_energy,
                "num_sites": len(selected_data)
            })
        except Exception as e:
            result["analysis_error"] = str(e)
    
    print(json.dumps(result))

if __name__ == '__main__':
    main()
