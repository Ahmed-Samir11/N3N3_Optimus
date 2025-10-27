# N3N3_Optimus — Multi-Warehouse Vehicle Routing Solvers

This repository contains iterative solver implementations and experiment artifacts for the
Beltone 2nd AI Hackathon (Robin Logistics). The codebase explores several solver
strategies for a Multi-Warehouse Vehicle Routing Problem (MWVRP) over a large road network.

Key contents
- `Ne3Na3_solver_1.py` ... `Ne3Na3_solver_8.py`: numbered solver iterations (standalone submissions).
- `ne3_quantum_prototype.py`: assignment-only QUBO prototype (order→vehicle) with expansion to routes.
- `ne3_qiskit_runner.py`: wrapper that imports the canonical Qiskit integration from `quantum/`.
- `quantum/`: quantum experiments, QAOA runner, Colab notebook and tests.
- `hackathon documents/`: helper scripts and test harnesses (headless runner, validator, dashboard helpers).

Quick start
1. Create and activate the provided virtual environment (Windows PowerShell):

```powershell
& .\optimus\Scripts\Activate.ps1
```

2. Run a headless solver test (fast validation):

```powershell
python -c "from robin_logistics import LogisticsEnvironment; from Ne3Na3_solver_7 import solver; env = LogisticsEnvironment(); result = solver(env); print(env.execute_solution(result))"
```

3. Run the QUBO assignment prototype (small experiments):

```powershell
python ne3_quantum_prototype.py
```

4. QAOA / quantum experiments
- Local Qiskit binary installs can be fragile on Windows (qiskit-aer). Use the notebook
  `quantum/colab_queuebo_qaoa.ipynb` for reproducible QAOA runs on Colab.
- Local fallback: the Qiskit runner includes a classical-sampler fallback so experiments
  can run without Qiskit primitives.

Experiment artifacts
- `ne3_quantum_prototype_results.csv`, `Ne3Na3_solver_84_results.csv`, and `comparison_with_fulfillment.csv`
  contain compact run metrics (seed, max_orders, runtime_s, valid_count, unfulfilled_count, exec_ok, cost).

Notes & conventions
- The solver contract for submissions is `def solver(env): return {"routes": [...]}`.
- Routes must include intermediate road-network nodes (not just pickups/deliveries).
- Per-run caching is allowed, but do not persist state between solver runs.

If you want me to archive or remove duplicate files (for example notebooks in both root and `quantum/`), tell me which copy to keep and I'll update the repo and commit the change.
