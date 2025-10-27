"""Run a tiny QAOA job on IBM Quantum via qiskit-ibm-runtime using your saved token.
This script assumes you've set your IBM token in the environment or saved it via
QiskitRuntimeService.save_account(...) locally.

It builds a very small random BQM (6 variables) and submits it through
`quantum.ne3_qiskit_runner.solve_qubo_with_qaoa(..., use_ibm_runtime=True)`.
"""
import random
import dimod
from quantum.ne3_qiskit_runner import solve_qubo_with_qaoa


def make_random_bqm(n=6, density=0.4):
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    for i in range(n):
        bqm.add_variable(f'x{i}', random.uniform(-1.0, 1.0))
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < density:
                bqm.add_interaction(f'x{i}', f'x{j}', random.uniform(-0.6, 0.6))
    return bqm


def main():
    print('Building tiny BQM and submitting to IBM runtime...')
    bqm = make_random_bqm(6, density=0.5)
    try:
        res = solve_qubo_with_qaoa(bqm, use_ibm_runtime=True, reps=1, shots=1024)
        print('QAOA result fval:', res.get('fval'))
        print('Assignment sample:', res.get('x'))
    except Exception as e:
        print('Hardware run failed:', e)


if __name__ == '__main__':
    main()
