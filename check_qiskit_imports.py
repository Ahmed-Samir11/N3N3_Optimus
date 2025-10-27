"""Thin wrapper that runs the import checks from quantum/check_qiskit_imports.py
so that the canonical implementation lives under the `quantum/` folder."""
from quantum.check_qiskit_imports import main


if __name__ == '__main__':
    main()
