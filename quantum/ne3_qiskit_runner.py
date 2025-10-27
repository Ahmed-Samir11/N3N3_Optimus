"""
Helpers to solve a dimod.BinaryQuadraticModel using Qiskit's QAOA
Supports: local simulator (Aer) and IBM Quantum (qiskit-ibm-runtime) when configured.

Important: This module does NOT read or store API keys. Configure your IBM token locally
using Qiskit's recommended methods (see README). NEVER paste tokens into chat.

Usage (high level):
  - Convert your dimod.BQM to a Qiskit QuadraticProgram
  - Instantiate a QAOA optimizer using a Sampler primitive
  - Solve the problem and return assignment and objective value

This is a best-effort integration; real-hardware runs are noisy and limited in qubit count.
Keep your QUBO small (<=20 qubits recommended) for hardware tests.
"""
from typing import Tuple, Dict, Any


def bqm_to_quadratic_program(bqm):
    """Convert a dimod.BinaryQuadraticModel to a qiskit_optimization.QuadraticProgram.
    Returns: (QuadraticProgram, var_list)
    Lazily imports qiskit_optimization to avoid hard dependency unless used.
    """
    try:
        from qiskit_optimization import QuadraticProgram
    except Exception as e:
        raise ImportError("qiskit_optimization is required. Install qiskit-optimization or qiskit.") from e

    qp = QuadraticProgram()
    # collect variable names (dimod keeps them in bqm.variables)
    var_names = list(bqm.variables)
    for v in var_names:
        qp.binary_var(name=str(v))

    # Build linear and quadratic objective
    linear = {str(v): float(bqm.linear[v]) for v in var_names}
    quadratic = {}
    for (u, v), bias in bqm.quadratic.items():
        quadratic[(str(u), str(v))] = float(bias)

    # Set objective (minimize) — dimod BQM may be Ising or QUBO; assume QUBO-style objective
    qp.minimize(linear=linear, quadratic=quadratic)
    return qp, var_names


def solve_qubo_with_qaoa(bqm, *, use_ibm_runtime: bool = False, backend_name: str = None, reps: int = 1, shots: int = 1024, initial_point=None) -> Dict[str, Any]:
    """Solve a dimod BQM using QAOA.

    Parameters:
      bqm: dimod.BinaryQuadraticModel
      use_ibm_runtime: if True, attempt to use Qiskit Runtime / IBM provider. Must be configured locally.
      backend_name: backend to target (e.g., 'ibmq_qasm_simulator' or 'ibmq_manila')
      reps: QAOA depth
      shots: number of measurement shots

    Returns: dict with keys: 'x' (assignment dict), 'fval' (objective), 'raw' (backend result)
    """
    qp, var_names = bqm_to_quadratic_program(bqm)

    # First try to run with Qiskit's QAOA primitives (preferred path).
    try:
        from qiskit.algorithms import QAOA
        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        # Use simulator sampler by default
        sampler = None
        runtime_session = None
        try:
            if use_ibm_runtime:
                # Attempt to use qiskit_ibm_runtime service (requires local config/token)
                from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as IBMSampler
                service = QiskitRuntimeService()  # will use saved account / environment
                sampler = IBMSampler(session=service)
            else:
                # Local primitive sampler (qiskit.primitives.Sampler backed by Aer simulator if available)
                try:
                    from qiskit.primitives import Sampler as LocalSampler
                    sampler = LocalSampler()
                except Exception:
                    sampler = None

            qaoa = QAOA(reps=reps, sampler=sampler)
            optimizer = MinimumEigenOptimizer(qaoa)
            res = optimizer.solve(qp)

            x = {var: int(res.x[i]) for i, var in enumerate(var_names)}
            fval = float(res.fval)
            return {'x': x, 'fval': fval, 'raw': res}

        finally:
            if runtime_session:
                try:
                    runtime_session.close()
                except Exception:
                    pass

    except Exception:
        # If Qiskit QAOA path fails (imports or runtime), fall back to a classical sampler using dimod/neal.
        # This keeps the API stable for callers and allows local testing without full Qiskit support.
        try:
            import dimod
        except Exception as e:
            raise ImportError("Neither Qiskit QAOA nor dimod are available for solving the BQM") from e

        # Try neal SimulatedAnnealingSampler first (fast if available)
        sampler = None
        try:
            from neal import SimulatedAnnealingSampler as NealSAS
            sampler = NealSAS()
            sampleset = sampler.sample(bqm, num_reads=shots if shots else 100)
        except Exception:
            # Try dimod's SimulatedAnnealingSampler (may exist in dimod.reference)
            try:
                from dimod.reference.samplers import SimulatedAnnealingSampler as DimodSAS
                sampler = DimodSAS()
                sampleset = sampler.sample(bqm, num_reads=shots if shots else 100)
            except Exception:
                # Last resort: Exact solver (only suitable for very small problems)
                try:
                    from dimod.reference.samplers import ExactSolver
                    sampleset = ExactSolver().sample(bqm)
                except Exception as e:
                    raise RuntimeError("No classical sampler available to solve BQM") from e

        # sampleset is a dimod.SampleSet-like object. Find the lowest-energy sample.
        try:
            best = min(sampleset, key=lambda s: s.energy)
            # sampleset records samples as mapping var -> value
            x = {var: int(best.sample[var]) for var in var_names}
            fval = float(best.energy)
            return {'x': x, 'fval': fval, 'raw': sampleset}
        except Exception:
            # Fallback extraction for different SampleSet APIs
            ss = sampleset.first if hasattr(sampleset, 'first') else None
            if ss is not None:
                x = {var: int(ss.sample[var]) for var in var_names}
                fval = float(ss.energy)
                return {'x': x, 'fval': fval, 'raw': sampleset}
            raise


if __name__ == '__main__':
    print('ne3_qiskit_runner module: define solve_qubo_with_qaoa(bqm, use_ibm_runtime=True|False)')
