import csv
from pathlib import Path
import ast

ROOT = Path(r"C:/Users/ahmed/OneDrive/Desktop/Beltone/N3N3_Optimus")
BASELINE = ROOT / "Ne3Na3_solver_84_results.csv"
PROTO = ROOT / "ne3_quantum_prototype_results.csv"
OUT = ROOT / "comparison_with_fulfillment.csv"


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_unfulfilled(s):
    if s is None or s == '':
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return list(v)
    except Exception:
        pass
    # fallback: empty
    return []


def main():
    with open(BASELINE, newline='', encoding='utf-8') as f:
        base_rows = list(csv.DictReader(f))
    with open(PROTO, newline='', encoding='utf-8') as f:
        proto_rows = list(csv.DictReader(f))

    proto_index = {(r['seed'], r['max_orders']): r for r in proto_rows}

    out_rows = []
    for b in base_rows:
        key = (b['seed'], b['max_orders'])
        p = proto_index.get(key, {})

        max_orders = None
        try:
            max_orders = int(b['max_orders'])
        except Exception:
            try:
                max_orders = int(p.get('max_orders'))
            except Exception:
                max_orders = None

        # prototype fulfillment
        proto_unf = []
        if 'unfulfilled' in p:
            proto_unf = parse_unfulfilled(p.get('unfulfilled', ''))
        proto_valid = None
        if max_orders is not None:
            proto_valid = None if proto_unf is None else (max_orders - len(proto_unf))
            proto_ful_pct = None if max_orders is None else (100.0 * (proto_valid or 0) / max_orders)
        else:
            proto_ful_pct = None

        # baseline fulfillment: prefer valid_count, else invalid_count
        base_valid = None
        if b.get('valid_count'):
            try:
                base_valid = int(b['valid_count'])
            except Exception:
                base_valid = None
        elif b.get('invalid_count'):
            try:
                invalid = int(b['invalid_count'])
                if max_orders is not None:
                    base_valid = max_orders - invalid
            except Exception:
                base_valid = None

        base_ful_pct = None
        if base_valid is not None and max_orders is not None:
            base_ful_pct = 100.0 * base_valid / max_orders

        proto_cost = to_float(p.get('cost'))
        base_cost = to_float(b.get('cost'))
        cost_delta = None
        pct_change = None
        if proto_cost is not None and base_cost is not None:
            cost_delta = proto_cost - base_cost
            pct_change = 100.0 * cost_delta / base_cost if base_cost != 0 else None

        out = {
            'seed': b.get('seed'),
            'max_orders': b.get('max_orders'),
            'proto_runtime_s': p.get('runtime_s',''),
            'baseline_runtime_s': b.get('runtime_s',''),
            'proto_exec_ok': p.get('exec_ok',''),
            'baseline_exec_ok': b.get('exec_ok',''),
            'proto_cost': p.get('cost',''),
            'baseline_cost': b.get('cost',''),
            'proto_ful_pct': '' if proto_ful_pct is None else f"{proto_ful_pct:.2f}",
            'baseline_ful_pct': '' if base_ful_pct is None else f"{base_ful_pct:.2f}",
            'cost_delta': '' if cost_delta is None else f"{cost_delta:.6f}",
            'percent_cost_change': '' if pct_change is None else f"{pct_change:.3f}"
        }
        out_rows.append(out)

    # write
    if out_rows:
        with open(OUT, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Wrote {OUT}")
        for r in out_rows:
            print(r)
    else:
        print('No rows to write')


if __name__ == '__main__':
    main()
