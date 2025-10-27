import importlib

mods = ['qiskit', 'qiskit_optimization', 'qiskit.algorithms', 'qiskit_optimization.algorithms']
for m in mods:
    try:
        importlib.import_module(m)
        print(m, 'OK')
    except Exception as e:
        print(m, 'IMPORT ERROR:', type(e).__name__, e)
