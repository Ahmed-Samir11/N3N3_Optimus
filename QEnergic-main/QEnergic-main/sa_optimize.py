# sa_optimize.py
"""
Simulated annealing QUBO solver with budget constraint using the 'simanneal' library.
Usage:
    python sa_optimize.py --qubo_file QUBO.json --budget 10000 --costs_file costs.json
QUBO.json format: {"Q": [[...]]}
costs.json format: [cost_0, cost_1, ...] (cost for each candidate)
Prints: {"selected_indices": [...], "fval": ..., "time_sec": ...}
"""
import argparse
import json
import numpy as np
import time
from simanneal import Annealer

class QUBOBudgetAnnealer(Annealer):
    def __init__(self, Q, costs, budget, state, record_progress=False, total_steps=10000, progress_interval=1):
        self.Q = Q
        self.costs = costs
        self.budget = budget
        self.record_progress = record_progress
        self.total_steps = total_steps
        self.progress_interval = progress_interval
        self.progress = []
        super(QUBOBudgetAnnealer, self).__init__(state)

    def move(self):
        # Flip a random bit
        n = len(self.state)
        i = np.random.randint(0, n)
        self.state[i] = 1 - self.state[i]

    def energy(self):
        x = np.array(self.state)
        # The energy is purely the QUBO value. All constraints (budget, etc.)
        # are already baked into the Q matrix. The extra penalty is removed.
        return float(x @ self.Q @ x)

    def update(self, step, T, E, acceptance, improvement):
        # Called by simanneal at each step
        if self.record_progress and (step % self.progress_interval == 0):
            x = np.array(self.state)
            total_cost = float(np.dot(x, self.costs))
            total_population = float(np.sum(x))
            self.progress.append({
                'step': step,
                'energy': float(x @ self.Q @ x),
                'total_cost': total_cost,
                'total_population': total_population,
                'solution': x.tolist()
            })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubo_file', type=str, required=True)
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--costs_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, help='Optional data file for analysis')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--tmax', type=float, default=25000.0, help='Initial temperature')
    parser.add_argument('--tmin', type=float, default=0.001, help='Final temperature')
    parser.add_argument('--record_progress', action='store_true', help='Record progress for visualization')
    parser.add_argument('--progress_interval', type=int, default=1, help='Record progress every N steps')
    args = parser.parse_args()
    with open(args.qubo_file, 'r') as f:
        Q = np.array(json.load(f)["Q"])
    with open(args.costs_file, 'r') as f:
        costs = np.array(json.load(f))
    t0 = time.time()
    
    # Start with a random initial state to improve exploration
    initial_state = np.random.randint(2, size=len(costs))
    
    annealer = QUBOBudgetAnnealer(Q, costs, args.budget, initial_state, record_progress=args.record_progress, total_steps=args.steps, progress_interval=args.progress_interval)
    annealer.steps = args.steps
    annealer.Tmax = args.tmax
    annealer.Tmin = args.tmin
    
    state, e = annealer.anneal()
    elapsed = time.time() - t0
    selected = [i for i, v in enumerate(state) if v]
    
    result = {"selected_indices": selected, "fval": e, "time_sec": elapsed}
    
    # Add analysis if data file is provided
    if args.data_file:
        try:
            with open(args.data_file, 'r') as f:
                data = json.load(f)
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
    
    # Save progress if requested
    if args.record_progress:
        with open('sa_progress.json', 'w') as f:
            json.dump(annealer.progress, f)
    
    print(json.dumps(result))

if __name__ == '__main__':
    main()
