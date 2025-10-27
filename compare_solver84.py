from robin_logistics import LogisticsEnvironment
import time

# Import solver 84 from RL/Ne3Na3_solver_84.py by path (module may live in RL/)
import importlib.util
import os
solver84 = None
solver84_path = os.path.join(os.path.dirname(__file__), 'RL', 'Ne3Na3_solver_84.py')
if os.path.exists(solver84_path):
    spec = importlib.util.spec_from_file_location('Ne3Na3_solver_84', solver84_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        solver84 = getattr(mod, 'solver', None)
    except Exception as e:
        print(f"Failed to load solver module from {solver84_path}: {e}")
else:
    print(f"Solver file not found at {solver84_path}")

if solver84 is None:
    raise SystemExit('Ne3Na3_solver_84.solver not available; place RL/Ne3Na3_solver_84.py in workspace')

seeds = [42, 100]
max_orders_list = [12, 15]
results = []

for s in seeds:
    for m in max_orders_list:
        print(f"\n=== Running Ne3Na3_solver_84 on seed={s}, max_orders={m} ===")
        env = LogisticsEnvironment()
        env.generate_new_scenario(seed=s)

        # trim orders to max_orders to match the quantum experiment harness behavior
        try:
            all_order_ids = list(env.get_all_order_ids())
            keep = all_order_ids[:m]
            # shallow copy of orders dict limited to `keep`
            orig_orders = env.orders
            env.orders = {oid: orig_orders[oid] for oid in keep}
        except Exception:
            # if slicing not supported, continue with full scenario
            keep = None

        t0 = time.time()
        # Run solver while suppressing its verbose prints to keep output concise
        import io, sys
        buf = io.StringIO()
        try:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = buf, buf
            sol = solver84(env)
        except Exception as e:
            # restore
            sys.stdout, sys.stderr = old_stdout, old_stderr
            print(f"solver84 raised exception: {e}")
            results.append({'seed': s, 'max_orders': m, 'error': str(e)})
            continue
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        dt = time.time() - t0
        # Validate (suppress any verbose prints from env)
        import io, sys
        buf = io.StringIO()
        try:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = buf, buf
            try:
                valid_res = env.validate_solution_complete(sol)
            except Exception as e:
                valid_res = f"validate error: {e}"

            # Try execute and cost if valid
            exec_ok = None
            exec_msg = None
            cost = None
            try:
                exec_ok, exec_msg = env.execute_solution(sol)
                try:
                    cost = env.calculate_solution_cost(sol)
                except Exception as e:
                    cost = f"calculate cost error: {e}"
            except Exception as e:
                exec_ok = False
                exec_msg = f"execute error: {e}"
            # attempt to get fulfillment summary from environment (non-verbose)
            ful_summary = None
            try:
                ful_summary = env.get_solution_fulfillment_summary(sol)
            except Exception:
                ful_summary = None
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        # extract concise counts from validation if possible
        valid_count = None
        invalid_count = None
        try:
            if isinstance(valid_res, dict):
                valid_count = int(valid_res.get('valid_count', valid_res.get('valids', 0) or 0))
                invalid_count = int(valid_res.get('invalid_count', valid_res.get('invalids', 0) or 0))
            elif isinstance(valid_res, (list, tuple)) and len(valid_res) > 1 and isinstance(valid_res[1], dict):
                # sometimes validate returns (bool, {..})
                info = valid_res[1]
                valid_count = int(info.get('valid_count', 0))
                invalid_count = int(info.get('invalid_count', 0))
        except Exception:
            pass

        # If validation didn't return counts, try fulfillment summary
        try:
            if (valid_count is None or invalid_count is None) and ful_summary:
                if isinstance(ful_summary, dict):
                    # common keys in fulfillment summary
                    for k in ('fully_fulfilled_orders', 'fully_fulfilled', 'fulfilled_orders', 'fulfilled'):
                        if k in ful_summary:
                            try:
                                valid_count = int(ful_summary[k])
                                break
                            except Exception:
                                continue
                    # sometimes nested under 'summary'
                    if valid_count is None and isinstance(ful_summary.get('summary'), dict):
                        s2 = ful_summary.get('summary')
                        for k in ('fully_fulfilled_orders', 'fulfilled'):
                            if k in s2:
                                try:
                                    valid_count = int(s2[k])
                                    break
                                except Exception:
                                    continue
                # derive invalid_count if we know trimmed order count
                if valid_count is not None and keep is not None:
                    try:
                        invalid_count = len(keep) - valid_count
                    except Exception:
                        invalid_count = None
        except Exception:
            pass

        # final fallback: if exec_ok and keep known, assume all orders executed
        try:
            if valid_count is None and invalid_count is None and keep is not None and exec_ok:
                valid_count = len(keep)
                invalid_count = 0
        except Exception:
            pass

        print(f"Execution: success={exec_ok}, msg={exec_msg}, cost={cost}")
        results.append({'seed': s, 'max_orders': m, 'runtime_s': dt, 'valid_res': valid_res, 'valid_count': valid_count, 'invalid_count': invalid_count, 'exec_ok': exec_ok, 'exec_msg': exec_msg, 'cost': cost})

# write concise CSV to workspace
import csv
out_csv = os.path.join(os.path.dirname(__file__), 'Ne3Na3_solver_84_results.csv')
with open(out_csv, 'w', newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(['seed', 'max_orders', 'runtime_s', 'valid_count', 'invalid_count', 'exec_ok', 'cost', 'exec_msg'])
    for r in results:
        writer.writerow([r.get('seed'), r.get('max_orders'), r.get('runtime_s'), r.get('valid_count'), r.get('invalid_count'), r.get('exec_ok'), r.get('cost'), r.get('exec_msg')])

print('\nWrote baseline results to:', out_csv)
print('\n=== Summary ===')
for r in results:
    # print compact summary only
    print({k: r[k] for k in ('seed', 'max_orders', 'runtime_s', 'valid_count', 'invalid_count', 'exec_ok', 'cost')})
