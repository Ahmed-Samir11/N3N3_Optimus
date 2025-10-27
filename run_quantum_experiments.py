from ne3_quantum_prototype import run_seed
import time
import csv
import os

seeds = [42, 100]
max_orders_list = [12, 15]
results = []

for s in seeds:
    for m in max_orders_list:
        print(f"\n>> Experiment seed={s}, max_orders={m}")
        t0 = time.time()
        try:
            res = run_seed(s, max_orders=m)
            ok = True
            err = None
        except Exception as e:
            res = None
            ok = False
            err = str(e)
            print(f"Run failed: {err}")
        dt = time.time() - t0
        # normalize result dict
        if isinstance(res, dict):
            row = {'seed': s, 'max_orders': m, 'runtime_s': dt, 'valid': res.get('valid'), 'unfulfilled': res.get('unfulfilled'), 'exec_ok': res.get('exec_ok'), 'exec_msg': res.get('exec_msg'), 'cost': res.get('cost'), 'error': res.get('error', err)}
        else:
            row = {'seed': s, 'max_orders': m, 'runtime_s': dt, 'valid': None, 'unfulfilled': None, 'exec_ok': None, 'exec_msg': None, 'cost': None, 'error': err}
        results.append(row)
        print(f"-- runtime: {dt:.2f}s, success: {ok}")

out_csv = os.path.join(os.path.dirname(__file__), 'ne3_quantum_prototype_results.csv')
with open(out_csv, 'w', newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(['seed', 'max_orders', 'runtime_s', 'exec_ok', 'cost', 'unfulfilled', 'error'])
    for r in results:
        writer.writerow([r['seed'], r['max_orders'], r['runtime_s'], r['exec_ok'], r['cost'], r['unfulfilled'], r['error']])

print(f"\nWrote prototype results to: {out_csv}")
print('\nSummary:')
for r in results:
    print({k: r[k] for k in ('seed', 'max_orders', 'runtime_s', 'exec_ok', 'cost')})
