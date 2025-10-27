"""Quick runner to test the Qiskit integration locally with a tiny random BQM.
This uses the local simulator (if qiskit/Aer installed). It does NOT contact IBM or require API keys.

Run:
  python quantum/run_qiskit_sim_test.py
"""
import random
import dimod
from quantum.ne3_qiskit_runner import solve_qubo_with_qaoa


def make_random_bqm(n=6, density=0.3):
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    for i in range(n):
        bqm.add_variable('x%d' % i, random.uniform(-1, 1))
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < density:
                bqm.add_interaction('x%d' % i, 'x%d' % j, random.uniform(-0.5, 0.5))
    return bqm


def main():
    bqm = make_random_bqm(6, density=0.5)
    print('BQM variables:', list(bqm.variables))
    try:
        res = solve_qubo_with_qaoa(bqm, use_ibm_runtime=False, reps=1, shots=512)
        print('Result fval:', res['fval'])
        print('Assignment:', res['x'])
    except Exception as e:
        print('QAOA run failed (local simulator may be missing qiskit/Aer):', e)


if __name__ == '__main__':
    main()
