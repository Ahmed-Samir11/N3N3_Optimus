"""ne3_quantum_prototype.py
Prototype: build a small assignment QUBO (order -> vehicle) for a pruned subset
of orders and solve it locally using dimod's SimulatedAnnealingSampler.

This script is conservative: it does NOT attempt to execute solutions in the
full `LogisticsEnvironment` (that requires correct pickup steps, inventory
bookkeeping and path expansion). Instead it produces feasible assignments and
an estimated cost based on shortest-path distances. Use this as a first
validation of the QUBO pipeline.
"""
from robin_logistics import LogisticsEnvironment
import dimod
from dimod import BinaryQuadraticModel
from dimod.reference.samplers import SimulatedAnnealingSampler
import networkx as nx
import numpy as np
import math
import time


def build_graph_from_env(env):
    rn = None
    try:
        rn = env.get_road_network_data()
    except Exception:
        rn = None

    G = nx.DiGraph()
    if not rn:
        return G

    # try adjacency_list or edges
    if isinstance(rn, dict) and 'adjacency_list' in rn:
        for u, nbrs in rn['adjacency_list'].items():
            # nbrs may be a list of neighbor ids or (neighbor, weight) pairs
            for nb in nbrs:
                try:
                    v, w = nb
                except Exception:
                    v = nb
                    # try to get weight via env.get_distance later, use 1.0 as fallback
                    w = 1.0
                G.add_edge(int(u), int(v), weight=float(w))
    elif isinstance(rn, dict) and 'edges' in rn:
        # edges likely (u,v,weight)
        for e in rn['edges']:
            if len(e) >= 3:
                u, v, w = e[0], e[1], e[2]
                G.add_edge(int(u), int(v), weight=float(w))
            elif len(e) == 2:
                u, v = e
                G.add_edge(int(u), int(v), weight=1.0)
    else:
        # attempt to iterate rn as list of edges
        try:
            for e in rn:
                if len(e) >= 3:
                    u, v, w = e[0], e[1], e[2]
                    G.add_edge(int(u), int(v), weight=float(w))
        except Exception:
            pass

    return G


def node_id_from_order(env, order):
    # order.destination may be a Node object with id
    try:
        return int(order.destination.id)
    except Exception:
        # fallback: try numeric index
        return int(getattr(order, 'destination', 0))


def vehicle_home_node(env, vehicle):
    # Try several common attributes used in this codebase
    try:
        if hasattr(vehicle, 'home_warehouse_id'):
            wh = env.warehouses.get(vehicle.home_warehouse_id)
            if wh is not None:
                return int(wh.location.id)
    except Exception:
        pass
    try:
        if hasattr(vehicle, 'home_warehouse'):
            wh = vehicle.home_warehouse
            return int(wh.location.id)
    except Exception:
        pass
    try:
        if hasattr(vehicle, 'home_node'):
            return int(vehicle.home_node)
    except Exception:
        pass
    # Last resort: try vehicle.node_id or id
    return int(getattr(vehicle, 'node_id', getattr(vehicle, 'id', 0)))


def build_assignment_bqm(env, orders, vehicles, G, k_candidates=3):
    # Precompute distances between vehicle homes and order nodes
    order_nodes = [node_id_from_order(env, env.orders[o_id]) for o_id in orders]
    vehicle_nodes = [vehicle_home_node(env, v) for v in vehicles]

    # compute shortest path lengths
    dist = {}
    for vi, vnode in enumerate(vehicle_nodes):
        for oi, onode in enumerate(order_nodes):
            try:
                d = nx.shortest_path_length(G, source=vnode, target=onode, weight='weight')
            except Exception:
                # fallback to env.get_distance or large number
                try:
                    d = env.get_distance(vnode, onode) or 1e6
                except Exception:
                    d = 1e6
            dist[(oi, vi)] = float(d)

    # Candidate pruning: for each order keep k closest vehicles
    candidates = {}
    # keep sorted lists as fallback if connectivity filtering removes all candidates
    sorted_vs_map = {}
    for oi in range(len(order_nodes)):
        sorted_vs = sorted(range(len(vehicle_nodes)), key=lambda vi: dist[(oi, vi)])
        sorted_vs_map[oi] = sorted_vs
        candidates[oi] = sorted_vs[:k_candidates]

    # Connectivity-based filtering: remove vehicles that cannot reach the order node
    # (prevents assignments that will produce infeasible routes during path expansion)
    if G is not None and G.number_of_nodes() > 0:
        for oi in range(len(order_nodes)):
            onode = order_nodes[oi]
            filtered = []
            for vi in candidates[oi]:
                vnode = vehicle_nodes[vi]
                try:
                    if vnode in G and onode in G and nx.has_path(G, vnode, onode):
                        filtered.append(vi)
                except Exception:
                    # if has_path fails for any reason, conservatively skip this vehicle
                    continue
            # if filtering removed all candidates, fall back to the closest vehicle(s)
            if filtered:
                candidates[oi] = filtered
            else:
                # keep at least the single nearest vehicle to allow solution construction
                candidates[oi] = sorted_vs_map[oi][:1]

    # Build BQM
    bqm = BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    # Variables x_o_v names
    x_names = {}
    for oi in range(len(order_nodes)):
        for vi in candidates[oi]:
            name = f"x_o{oi}_v{vi}"
            x_names[(oi, vi)] = name
            # linear cost = distance
            lin = dist[(oi, vi)]
            bqm.add_variable(name, lin)

    # vehicle open vars y_v
    y_names = {}
    fixed_cost = 300.0
    for vi in range(len(vehicle_nodes)):
        y = f"y_v{vi}"
        y_names[vi] = y
        bqm.add_variable(y, fixed_cost)

    # One-hot constraint per order: A*(sum x -1)^2
    max_lin = max([abs(v) for v in bqm.linear.values()]) if bqm.linear else 1.0
    A = max(1000.0, 10 * max_lin)
    for oi in range(len(order_nodes)):
        vars_oi = [x_names[(oi, vi)] for vi in candidates[oi]]
        # expand (sum vars)^2
        for v in vars_oi:
            bqm.add_linear(v, A * 1.0)  # from sum^2 term
        for i in range(len(vars_oi)):
            for j in range(i+1, len(vars_oi)):
                bqm.add_quadratic(vars_oi[i], vars_oi[j], 2 * A)
        # and the -2*A*sum x term and +A constant; implement -2A via linear
        for v in vars_oi:
            bqm.add_linear(v, -2 * A)
        # constant A added to offset (ignored)

    # Link x and y: penalize x*(1-y) with penalty C => +C*x - C*x*y
    C = A * 10
    for (oi, vi), name in x_names.items():
        bqm.add_linear(name, C)
        bqm.add_quadratic(name, y_names[vi], -C)

    return bqm, x_names, y_names, candidates, dist


def solve_bqm_and_extract(bqm, x_names, y_names, candidates, dist, num_reads=200):
    sampler = SimulatedAnnealingSampler()
    sample_set = sampler.sample(bqm, num_reads=num_reads)
    best = None
    best_energy = float('inf')
    best_sample = None
    for record in sample_set.data(['sample', 'energy']):
        sample = record.sample
        energy = record.energy
        # compute constraint violations (one-hot)
        viol = 0
        # iterate over actual order indices in candidates
        for oi in candidates.keys():
            s = sum(sample[x_names[(oi, vi)]] for vi in candidates[oi])
            viol += abs(s - 1)
        # prefer feasible with lower energy
        if viol == 0 and energy < best_energy:
            best_energy = energy
            best_sample = sample
            break
        if best_sample is None and energy < best_energy:
            best_energy = energy
            best_sample = sample

    return best_sample, best_energy


def pretty_print_assignment(best_sample, orders, vehicles, candidates, dist):
    assign = {}
    total_cost = 0.0
    for oi_idx, o_id in enumerate(orders):
        assigned = None
        for vi in candidates[oi_idx]:
            var = f"x_o{oi_idx}_v{vi}"
            if best_sample.get(var, 0) == 1:
                assigned = vi
                break
        if assigned is None:
            print(f"Order {o_id} unassigned in sample")
        else:
            v = vehicles[assigned]
            assign[o_id] = v.id
            total_cost += dist[(oi_idx, assigned)]

    print("\nAssignment result:")
    for o_id, vid in assign.items():
        print(f"  Order {o_id} -> Vehicle {vid}")
    print(f"Estimated total distance (sum d(home,order)): {total_cost:.2f}")


def build_executable_solution(env, best_sample, orders, vehicles, candidates, dist, G):
    """Convert assignment sample into a solution dict that the environment can validate.
    We'll attempt to allocate inventory from warehouses (allowing per-sku splits) and
    build naive routes: home -> [warehouses...] -> order -> home.
    Returns (solution_dict, unfulfilled_orders_list, validation_message)
    """
    # copy warehouse inventory
    wh_inv = {wh_id: env.get_warehouse_inventory(wh_id).copy() for wh_id in env.warehouses.keys()}

    # map warehouse nodes
    wh_node = {}
    for wh_id, wh in env.warehouses.items():
        try:
            wh_node[wh_id] = int(wh.location.id)
        except Exception:
            wh_node[wh_id] = int(getattr(wh.location, 'id', 0))

    routes = []
    unfulfilled = []

    # first, group orders by assigned vehicle index
    assigned_orders = {}
    for oi_idx, o_id in enumerate(orders):
        assigned_vi = None
        for vi in candidates[oi_idx]:
            var = f"x_o{oi_idx}_v{vi}"
            if best_sample.get(var, 0) == 1:
                assigned_vi = vi
                break
        if assigned_vi is None:
            unfulfilled.append(o_id)
            continue
        assigned_orders.setdefault(assigned_vi, []).append((oi_idx, o_id))

    # build a single route per vehicle with all its assigned orders
    for vi, order_list in assigned_orders.items():
        vehicle = vehicles[vi]
        home_node = vehicle_home_node(env, vehicle)
        steps = []
        steps.append({"node_id": int(home_node), "pickups": [], "deliveries": [], "unloads": []})
        current = home_node

        for oi_idx, o_id in order_list:
            order = env.orders[o_id]
            try:
                order_node = int(order.destination.id)
            except Exception:
                order_node = int(getattr(order, 'destination', 0))
            requirements = env.get_order_requirements(o_id)

            # allocate per-sku across warehouses (allow splits)
            pickups_by_wh = {}
            allocation_ok = True
            # order warehouses sorted by distance from current
            whs_sorted = sorted(env.warehouses.keys(), key=lambda wid: (
                nx.shortest_path_length(G, source=current, target=wh_node[wid], weight='weight')
                if G and current in G and wh_node[wid] in G else float('inf')
            ))

            for sku_id, qty in requirements.items():
                remaining = qty
                for wid in whs_sorted:
                    avail = wh_inv.get(wid, {}).get(sku_id, 0)
                    if avail <= 0:
                        continue
                    take = min(avail, remaining)
                    pickups_by_wh.setdefault(wid, {})
                    pickups_by_wh[wid][sku_id] = pickups_by_wh[wid].get(sku_id, 0) + take
                    wh_inv[wid][sku_id] = wh_inv[wid].get(sku_id, 0) - take
                    remaining -= take
                    if remaining <= 0:
                        break
                if remaining > 0:
                    allocation_ok = False
                    break

            if not allocation_ok:
                unfulfilled.append(o_id)
                continue

            # visit warehouses needed for this order
            wh_visit_order = [wid for wid in whs_sorted if wid in pickups_by_wh]
            for wid in wh_visit_order:
                target_node = wh_node[wid]
                # ensure reachability on directed graph before adding path
                if G is not None and current in G and target_node in G and nx.has_path(G, current, target_node):
                    path = nx.shortest_path(G, source=current, target=target_node, weight='weight')
                else:
                    # cannot reach required warehouse from current node: mark this order unfulfilled
                    allocation_ok = False
                    break
                for n in path[1:]:
                    steps.append({"node_id": int(n), "pickups": [], "deliveries": [], "unloads": []})
                pk_list = []
                for sku_id, q in pickups_by_wh[wid].items():
                    pk_list.append({"warehouse_id": wid, "sku_id": sku_id, "quantity": int(q)})
                steps[-1]["pickups"] = pk_list
                current = target_node

            # go to order node and deliver (ensure reachability)
            if G is not None and current in G and order_node in G and nx.has_path(G, current, order_node):
                path = nx.shortest_path(G, source=current, target=order_node, weight='weight')
            else:
                allocation_ok = False
                break
            for n in path[1:]:
                steps.append({"node_id": int(n), "pickups": [], "deliveries": [], "unloads": []})
            dl_list = []
            for sku_id, qty in requirements.items():
                dl_list.append({"order_id": o_id, "sku_id": sku_id, "quantity": int(qty)})
            steps[-1]["deliveries"] = dl_list
            current = order_node

        # after all assigned orders, return home (ensure reachability)
        if G is not None and current in G and home_node in G and nx.has_path(G, current, home_node):
            path = nx.shortest_path(G, source=current, target=home_node, weight='weight')
            for n in path[1:]:
                steps.append({"node_id": int(n), "pickups": [], "deliveries": [], "unloads": []})
        else:
            # if vehicle cannot return home via network, mark route invalid by skipping this route
            # (leave allocation as is and let env.validate catch it if needed)
            pass

        # ensure route starts and ends at home_node
        try:
            route_ok = (steps[0]["node_id"] == int(home_node) and steps[-1]["node_id"] == int(home_node))
        except Exception:
            route_ok = False
        if not route_ok:
            # mark all orders in this route as unfulfilled so they won't appear as assigned
            for _, o_id in order_list:
                if o_id not in unfulfilled:
                    unfulfilled.append(o_id)
            # skip adding this invalid route
        else:
            routes.append({"vehicle_id": vehicle.id, "steps": steps})

    solution = {"routes": routes}

    # validate
    try:
        res = env.validate_solution_complete(solution)
        # support variable-length returns
        if isinstance(res, tuple) or isinstance(res, list):
            if len(res) == 0:
                valid = False
                msg = "validate_solution_complete returned empty"
            else:
                valid = bool(res[0])
                msg = " | ".join(str(x) for x in res[1:]) if len(res) > 1 else ""
        else:
            # single boolean-like
            valid = bool(res)
            msg = ""
    except Exception as e:
        valid = False
        msg = f"validate_solution_complete error: {e}"

    return solution, unfulfilled, (valid, msg)


def run_seed(seed, max_orders=10):
    print(f"\n=== Running seed {seed} (max_orders={max_orders}) ===")
    t0 = time.time()
    env = LogisticsEnvironment()
    env.generate_new_scenario(seed=seed)

    all_order_ids = list(env.get_all_order_ids())
    orders = all_order_ids[:max_orders]

    vehicles = list(env.get_all_vehicles())

    G = build_graph_from_env(env)

    bqm, x_names, y_names, candidates, dist = build_assignment_bqm(env, orders, vehicles, G, k_candidates=3)

    best_sample, energy = solve_bqm_and_extract(bqm, x_names, y_names, candidates, dist)
    if best_sample is None:
        print("No sample found")
        return {'seed': seed, 'max_orders': max_orders, 'runtime_s': time.time() - t0, 'error': 'no_sample'}

    pretty_print_assignment(best_sample, orders, vehicles, candidates, dist)

    # build executable solution and validate
    solution, unfulfilled, (valid, msg) = build_executable_solution(env, best_sample, orders, vehicles, candidates, dist, G)
    print(f"\nValidation: valid={valid}, msg={msg}")
    if unfulfilled:
        print(f"Unfulfilled orders due to inventory: {unfulfilled}")

    exec_ok = None
    exec_msg = None
    cost = None
    if valid:
        try:
            exec_ok, exec_msg = env.execute_solution(solution)
            cost = env.calculate_solution_cost(solution)
            print(f"Execution: success={exec_ok}, msg={exec_msg}, cost={cost}")
        except Exception as e:
            exec_ok = False
            exec_msg = f"execute error: {e}"
            print(f"Error executing solution: {e}")

    runtime = time.time() - t0
    return {'seed': seed, 'max_orders': max_orders, 'runtime_s': runtime, 'valid': (valid, msg), 'unfulfilled': unfulfilled, 'exec_ok': exec_ok, 'exec_msg': exec_msg, 'cost': cost}


if __name__ == '__main__':
    seeds = [42, 100]
    for s in seeds:
        # run with slightly larger pruned subsets to test scaling
        run_seed(s, max_orders=12)
        run_seed(s, max_orders=15)
