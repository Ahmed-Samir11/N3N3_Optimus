"""
Helpers to solve a dimod.BinaryQuadraticModel using Qiskit's QAOA
Supports: local simulator (Aer) and IBM Quantum (qiskit-ibm-runtime) when configured.

Important: This module does NOT read or store API keys. Configure your IBM token locally
using Qiskit's recommended methods (see README below). NEVER paste tokens into chat.

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
    """Top-level wrapper that re-exports the Qiskit runner implementation from the
    `quantum` subfolder. This keeps quantum-related code centralized while
    preserving backward-compatible imports for existing scripts.

    Do not place heavy Qiskit imports here; they belong in `quantum/ne3_qiskit_runner.py`.
    """
    from quantum.ne3_qiskit_runner import bqm_to_quadratic_program, solve_qubo_with_qaoa


    __all__ = ["bqm_to_quadratic_program", "solve_qubo_with_qaoa"]


    if __name__ == '__main__':
        print('ne3_qiskit_runner wrapper: import solve_qubo_with_qaoa from quantum.ne3_qiskit_runner')

