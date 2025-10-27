# tabu_search_optimize.py
"""
Tabu Search QUBO solver.
Usage:
    python tabu_search_optimize.py --qubo_file QUBO.json --data_file data.json --iterations 1000 --tenure 10
"""
import argparse
import json
import numpy as np
import time
from collections import deque

def tabu_search(Q, num_sites, iterations=1000, tenure=10, initial_state=None):
    if initial_state is None:
        current_solution = np.random.randint(2, size=num_sites)
    else:
        current_solution = np.array(initial_state)

    best_solution = current_solution.copy()
    best_energy = float(current_solution @ Q @ current_solution)
    
    tabu_list = deque(maxlen=tenure)

    for it in range(iterations):
        neighborhood_best_energy = float('inf')
        best_neighbor = None
        move_to_make = -1

        for i in range(num_sites):
            if i in tabu_list:
                continue  
            neighbor = current_solution.copy()
            neighbor[i] = 1 - neighbor[i]
            neighbor_energy = float(neighbor @ Q @ neighbor)

            if neighbor_energy < neighborhood_best_energy:
                neighborhood_best_energy = neighbor_energy
                best_neighbor = neighbor
                move_to_make = i

        if best_neighbor is not None:
            current_solution = best_neighbor
            tabu_list.append(move_to_make)
            
            if neighborhood_best_energy < best_energy:
                best_solution = current_solution.copy()
                best_energy = neighborhood_best_energy
    
    return best_solution, best_energy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubo_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, help='Optional data file for analysis')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--tenure', type=int, default=10, help='Tabu tenure (size of tabu list)')
    args = parser.parse_args()

    with open(args.qubo_file, 'r') as f:
        Q = np.array(json.load(f)["Q"])
    
    num_sites = Q.shape[0]
    
    t0 = time.time()
    solution_vector, energy = tabu_search(Q, num_sites, args.iterations, args.tenure)
    elapsed = time.time() - t0

    selected = [i for i, v in enumerate(solution_vector) if v]
    result = {"selected_indices": selected, "fval": energy, "time_sec": elapsed}

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
    
    print(json.dumps(result))

if __name__ == '__main__':
    main() 