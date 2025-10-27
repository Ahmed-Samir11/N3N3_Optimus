# Quantum experiments (QAOA) — Notes and development guide

This folder centralizes quantum-related artifacts: QAOA runners, Qiskit integration tests,
and the Colab demonstration notebook. It was created to keep quantum development separate
from the main logistics solver code.

Purpose
- Host the QUBO→QAOA integration prototypes used to explore quantum-assisted assignment strategies.
- Provide a Colab-friendly notebook (pinned Qiskit versions) for reproducible runs on cloud hardware.
- Keep local development working with classical fallbacks when Qiskit binaries are unavailable.

Important constraints and notes
- Baseline solver file (Ne3Na3_solver_84.py) is preserved unchanged. All comparisons were done by
  pruning orders and running both the baseline and the assignment-prototype under identical seeds.
- The logistics environment enforces that routes include intermediate path nodes. The prototype
  builds assignments (order → vehicle) and then expands them into full routes using Dijkstra paths
  (per-run caches only).
- Fulfillment-first policy: achieving 100% fulfillment is prioritized over reducing cost.

Local environment portability
- Qiskit (especially qiskit-aer) has binary dependencies that are fragile on some Windows hosts.
  If you cannot install qiskit-aer, the runner falls back to classical samplers (neal / dimod).
- For reproducible QAOA runs on hardware or the full Qiskit stack, use Colab or a Linux/Conda environment
  where Qiskit binary wheels are available.

Files
- ne3_qiskit_runner.py — canonical runner implementing BQM→QuadraticProgram, QAOA invocation, and
  a classical fallback (when Qiskit primitives are missing).
- run_qiskit_sim_test.py — small simulation test that uses the runner; has a wrapper at repo root.
- run_qiskit_hw_test.py — small hardware test (IBM runtime); has a wrapper at repo root.
- check_qiskit_imports.py — quick import diagnostics to check installed Qiskit subpackages.
- colab_queuebo_qaoa.ipynb — Colab notebook with pinned install and a working QueueBo QAOA demo.

Development & reproduction notes
1. To run headless logistic tests locally (fast validation):
   python -c "from robin_logistics import LogisticsEnvironment; from Ne3Na3_solver_7 import solver; env = LogisticsEnvironment(); result = solver(env); print(env.execute_solution(result))"

2. To run the QAOA demo on Colab:
   - Open `colab_queuebo_qaoa.ipynb` in Colab and run the cells. The notebook installs a compatible
     Qiskit stack and runs a small QAOA demonstration.

3. If you want to iterate on the QUBO or QAOA parameters, edit `quantum/ne3_qiskit_runner.py`.

Misc
- Keep per-run caches inside solver calls only (do not persist between runs).
- Avoid printing large route-validation objects to stdout; write structured CSVs instead (the prototype
  already emits concise CSV metrics).

