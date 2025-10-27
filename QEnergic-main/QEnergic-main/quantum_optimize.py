# quantum_optimize.py
"""
Quantum annealing QUBO solver using Qiskit (simulated quantum annealing via QAOA).
This script expects a QUBO matrix and parameters as input, and returns the selected grid indices and timing.

Usage:
    python quantum_optimize.py --qubo_file QUBO.json --k 3

QUBO.json format:
    [[Q00, Q01, ...], [Q10, Q11, ...], ...]

Returns:
    Prints selected indices and time taken.
"""
import argparse
import json
import time
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit.primitives import Sampler
from qiskit import Aer
from qiskit.utils import algorithm_globals


def load_qubo(file_path):
    with open(file_path, 'r') as f:
        Q = json.load(f)
    return np.array(Q)

def build_quadratic_program(Q, k):
    n = Q.shape[0]
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f'x{i}')
    # Objective: sum_i sum_j Q[i][j] x_i x_j
    linear = {f'x{i}': float(Q[i, i]) for i in range(n)}
    quadratic = {(f'x{i}', f'x{j}'): float(Q[i, j]) for i in range(n) for j in range(n) if i != j and Q[i, j] != 0}
    qp.minimize(linear=linear, quadratic=quadratic)
    # Constraint: sum x_i == k
    qp.linear_constraint(linear={f'x{i}': 1 for i in range(n)}, sense='==', rhs=k, name='select_k')
    return qp

def solve_qubo_qaoa(Q, k, reps=2, seed=42):
    qp = build_quadratic_program(Q, k)
    algorithm_globals.random_seed = seed
    backend = Aer.get_backend('aer_simulator_statevector')
    qaoa = QAOA(sampler=Sampler(), reps=reps, seed=seed)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)
    x = [int(result.x[i]) for i in range(len(result.x))]
    return x, result.fval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubo_file', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--reps', type=int, default=2)
    args = parser.parse_args()
    Q = load_qubo(args.qubo_file)
    start = time.time()
    x, fval = solve_qubo_qaoa(Q, args.k, reps=args.reps)
    elapsed = time.time() - start
    print(json.dumps({'selected_indices': [i for i, v in enumerate(x) if v], 'fval': fval, 'time_sec': elapsed}))

if __name__ == '__main__':
    main()
