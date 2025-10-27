"""Run additional prototype cases (seeds and max_orders) and append results to the prototype CSV.
This imports run_seed from ne3_quantum_prototype to avoid duplicating logic.
"""
from pathlib import Path
import csv
import time

ROOT = Path(r"C:/Users/ahmed/OneDrive/Desktop/Beltone/N3N3_Optimus")
PROTO_CSV = ROOT / "ne3_quantum_prototype_results.csv"

def append_row(row):
    fieldnames = ['seed','max_orders','runtime_s','valid_count','invalid_count','exec_ok','cost','exec_msg']
    exists = PROTO_CSV.exists()
    with open(PROTO_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        # ensure only the expected keys
        out = {k: row.get(k, '') for k in fieldnames}
        writer.writerow(out)

def main():
    # Defer import so the module can be edited without circulars
    from ne3_quantum_prototype import run_seed

    seeds = [7, 13]
    max_orders_list = [12, 15]

    for seed in seeds:
        for m in max_orders_list:
            print(f"Running prototype seed={seed}, max_orders={m}")
            t0 = time.time()
            res = run_seed(seed=seed, max_orders=m)
            runtime = time.time() - t0
            # normalize result dict keys to match CSV
            row = {
                'seed': str(seed),
                'max_orders': str(m),
                'runtime_s': f"{runtime:.6f}",
                'valid_count': res.get('valid_count', res.get('valid', '')),
                'invalid_count': res.get('invalid_count', res.get('invalid', '')),
                'exec_ok': res.get('exec_ok', ''),
                'cost': res.get('cost', ''),
                'exec_msg': res.get('exec_msg', '')
            }
            append_row(row)
            print(f" -> wrote result for seed={seed}, max_orders={m}")

if __name__ == '__main__':
    main()
