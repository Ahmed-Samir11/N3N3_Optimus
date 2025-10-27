import csv
from pathlib import Path

ROOT = Path(r"C:/Users/ahmed/OneDrive/Desktop/Beltone/N3N3_Optimus")
BASELINE_CSV = ROOT / "Ne3Na3_solver_84_results.csv"
PROTO_CSV = ROOT / "ne3_quantum_prototype_results.csv"
OUT_CSV = ROOT / "comparison_table.csv"


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def as_float(x):
    if x is None:
        return None
    if isinstance(x, float):
        return x
    s = str(x).strip()
    if s == '' or s.lower() == 'none':
        return None
    try:
        return float(s)
    except Exception:
        return None


def main():
    baseline = read_csv(BASELINE_CSV)
    proto = read_csv(PROTO_CSV)

    # index proto by (seed,max_orders)
    proto_index = {(p['seed'], p['max_orders']): p for p in proto}

    rows = []
    for b in baseline:
        key = (b['seed'], b['max_orders'])
        p = proto_index.get(key, {})
        proto_cost = as_float(p.get('cost'))
        base_cost = as_float(b.get('cost'))
        delta = None
        pct = None
        if proto_cost is not None and base_cost is not None:
            delta = proto_cost - base_cost
            pct = 100.0 * delta / base_cost if base_cost != 0 else None

        row = {
            'seed': b.get('seed'),
            'max_orders': b.get('max_orders'),
            'proto_runtime_s': p.get('runtime_s', ''),
            'baseline_runtime_s': b.get('runtime_s', ''),
            'proto_exec_ok': p.get('exec_ok', ''),
            'baseline_exec_ok': b.get('exec_ok', ''),
            'proto_cost': p.get('cost', ''),
            'baseline_cost': b.get('cost', ''),
            'cost_delta': '' if delta is None else f"{delta:.6f}",
            'percent_cost_change': '' if pct is None else f"{pct:.3f}"
        }
        rows.append(row)

    # write out
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote comparison CSV: {OUT_CSV}")
    print("Summary table:")
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
