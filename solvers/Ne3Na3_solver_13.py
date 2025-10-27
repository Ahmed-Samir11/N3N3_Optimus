import random
import time
from typing import Any, Dict, List, Tuple
from robin_logistics import LogisticsEnvironment
import heapq

# Lightweight memetic solver: population of solutions + local search (LNS-like)
# This solver is self-contained and uses env APIs. It's a compact, pragmatic implementation
# tuned to produce valid solutions quickly rather than optimal.

def compute_route_distance(env: Any, steps: List[Dict]) -> float:
    # Use env.get_route_distance if available, else sum pairwise distances
    try:
        route = {"vehicle_id": steps[0]["vehicle_id"] if steps else None, "steps": steps}
        return env.get_route_distance(route)
    except Exception:
        total = 0.0
        prev = steps[0]["node_id"] if steps else None
        for s in steps[1:]:
            try:
                d = env.get_distance(prev, s["node_id"]) or 0.0
            except Exception:
                d = 0.0
            total += d
            prev = s["node_id"]
        return total


def dijkstra_shortest_path(env: Any, start: int, end: int) -> Tuple[List[int], float]:
    if start == end:
        return [start], 0.0
    try:
        road = env.get_road_network_data()
        adjacency = road.get("adjacency_list", {})
    except Exception:
        adjacency = {}
    if not adjacency or (start not in adjacency and end not in adjacency):
        d = env.get_distance(start, end)
        if d is None:
            return None, None
        return [start, end], float(d)

    dist = {start: 0.0}
    prev = {}
    heap = [(0.0, start)]
    visited = set()
    while heap:
        cur_d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            break
        for nb in adjacency.get(node, []):
            try:
                w = env.get_distance(node, nb)
                if w is None:
                    continue
                nd = cur_d + float(w)
            except Exception:
                continue
            if nb not in dist or nd < dist[nb]:
                dist[nb] = nd
                prev[nb] = node
                heapq.heappush(heap, (nd, nb))
    if end not in dist:
        return None, None
    path = []
    cur = end
    while True:
        path.append(cur)
        if cur == start:
            break
        cur = prev.get(cur)
        if cur is None:
            return None, None
    path.reverse()
    return path, dist[end]


def build_steps_with_path(env: Any, node_sequence: List[Tuple[int, Any]], home_node: int) -> List[Dict]:
    """Given a sequence of (node_id, payload) where payload may be:
      - a list: interpreted as deliveries for that node, or
      - a dict: may contain 'pickups' and/or 'deliveries' lists.
    Expand each leg into intermediate path nodes and return a steps list suitable for the env.
    node_sequence should be a list of tuples (node_id, payload) excluding the starting home node.
    The returned steps begin with the home_node step.
    """
    steps = []
    steps.append({"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []})
    current = home_node
    for node_id, deliveries in node_sequence:
        path, _ = dijkstra_shortest_path(env, current, node_id)
        if path is None:
            return None
        # normalize payload into pickups/deliveries
        if isinstance(deliveries, dict):
            pickups_list = deliveries.get("pickups", []) or []
            deliveries_list = deliveries.get("deliveries", []) or []
        else:
            pickups_list = []
            deliveries_list = deliveries or []
        # include intermediate nodes (excluding current, include node_id)
        if len(path) <= 1:
            # node is the same as current: attach pickups/deliveries to current step
            last = steps[-1]
            # merge pickups
            if pickups_list:
                last_pick = last.get("pickups", []) or []
                # append pickups
                last["pickups"] = last_pick + pickups_list
            if deliveries_list:
                last_del = last.get("deliveries", []) or []
                last["deliveries"] = last_del + deliveries_list
        else:
            for intermediate in path[1:]:
                if intermediate == node_id:
                    # delivery node: include deliveries and pickups (if any)
                    steps.append({"node_id": intermediate, "pickups": pickups_list, "deliveries": deliveries_list, "unloads": []})
                else:
                    steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
        current = node_id
    # finally, path back to home
    path_home, _ = dijkstra_shortest_path(env, current, home_node)
    if path_home is None:
        return None
    for intermediate in path_home[1:]:
        steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
    return steps


def make_solution_empty() -> Dict[str, List]:
    return {"routes": []}


def solution_cost(env: Any, solution: Dict) -> float:
    try:
        return env.calculate_solution_cost(solution)
    except Exception:
        # fallback to sum of route distances
        total = 0.0
        for r in solution.get("routes", []):
            steps = r.get("steps", [])
            if not steps:
                continue
            prev = steps[0]["node_id"]
            for step in steps[1:]:
                total += env.get_distance(prev, step["node_id"]) or 0.0
                prev = step["node_id"]
        return total


def greedy_initial_solution(env: Any, shipments_by_wh: Dict[str, Dict[str, Dict[str, int]]]) -> Dict:
    """Very similar to earlier greedy: fill vehicles per warehouse with select_orders_for_vehicle style selection.
    This implementation prioritizes feasibility and speed.
    """
    solution = {"routes": []}
    warehouses = env.warehouses

    # group vehicles by warehouse
    vehicles_by_wh = {}
    for v in env.get_all_vehicles():
        home_wh = getattr(v, "home_warehouse_id", None)
        if home_wh is None:
            try:
                hn = env.get_vehicle_home_warehouse(v.id)
                # find matching wh id
                found = None
                for wid, w in warehouses.items():
                    wnode = getattr(w.location, "id", w.location)
                    if wnode == hn:
                        found = wid
                        break
                home_wh = found or (list(warehouses.keys())[0] if warehouses else None)
            except Exception:
                home_wh = list(warehouses.keys())[0] if warehouses else None
        vehicles_by_wh.setdefault(home_wh, []).append(v)

    for wh_id, wh_ship in shipments_by_wh.items():
        if not wh_ship:
            continue
        wh_obj = warehouses[wh_id]
        wh_node = getattr(wh_obj.location, "id", wh_obj.location)
        remaining = {oid: dict(smap) for oid, smap in wh_ship.items()}
        for vehicle in vehicles_by_wh.get(wh_id, []):
            # simple greedy: assign single orders until capacity
            try:
                rem_w, rem_v = env.get_vehicle_remaining_capacity(vehicle.id)
            except Exception:
                rem_w, rem_v = getattr(vehicle, "capacity_weight", 0.0), getattr(vehicle, "capacity_volume", 0.0)
            assigned = {}
            for oid, smap in list(remaining.items()):
                # compute weight/volume
                w = 0.0
                v = 0.0
                for sku, q in smap.items():
                    sku_obj = env.skus[sku]
                    w += sku_obj.weight * q
                    v += sku_obj.volume * q
                if w <= rem_w + 1e-9 and v <= rem_v + 1e-9:
                    assigned[oid] = dict(smap)
                    rem_w -= w
                    rem_v -= v
                    remaining.pop(oid, None)
            if not assigned:
                continue
            # build node sequence (delivery nodes) and expand with path
            try:
                home_node = env.get_vehicle_home_warehouse(vehicle.id)
            except Exception:
                home_node = getattr(wh_obj.location, "id", wh_obj.location)
            node_sequence = []
            # warehouse step: include pickups so vehicle loads items before deliveries
            # Always include the pickup node in the sequence; build_steps_with_path will merge it
            # into the starting step when node == home_node.
            pickup_map = {}
            for oid, smap in assigned.items():
                for sku, q in smap.items():
                    pickup_map[sku] = pickup_map.get(sku, 0) + int(q)
            # include warehouse_id so the environment can map the pickup to a warehouse inventory
            pickups_list = [{"warehouse_id": wh_id, "sku_id": sku, "quantity": int(q)} for sku, q in pickup_map.items()]
            node_sequence.append((wh_node, {"pickups": pickups_list}))
            for oid in assigned.keys():
                node = env.get_order_location(oid)
                deliveries = [{"order_id": oid, "sku_id": sku, "quantity": int(q)} for sku, q in assigned[oid].items()]
                node_sequence.append((node, deliveries))
            steps = build_steps_with_path(env, node_sequence, home_node)
            if steps is not None:
                solution["routes"].append({"vehicle_id": vehicle.id, "steps": steps})
    return solution


# Minimal memetic algorithm

def mutate_solution(env: Any, solution: Dict, shipments_by_wh: Dict) -> Dict:
    # random remove a few orders and try to reinsert greedily
    all_orders = []
    for r in solution.get("routes", []):
        for s in r.get("steps", []):
            for d in s.get("deliveries", []):
                all_orders.append(d["order_id"])
    if not all_orders:
        return solution
    to_remove = set(random.sample(all_orders, min(len(all_orders), max(1, int(0.05 * len(all_orders))))))
    # remove
    new_routes = []
    removed_items = {}
    for r in solution.get("routes", []):
        new_steps = []
        for step in r.get("steps", []):
            new_del = []
            for d in step.get("deliveries", []):
                if d["order_id"] in to_remove:
                    removed_items.setdefault(d["order_id"], {})
                    removed_items[d["order_id"]].setdefault(d["sku_id"], 0)
                    removed_items[d["order_id"]][d["sku_id"]] += int(d["quantity"])
                else:
                    new_del.append(d)
            new_steps.append({"node_id": step["node_id"], "pickups": step.get("pickups", []), "deliveries": new_del, "unloads": step.get("unloads", [])})
        if any(st.get("deliveries") for st in new_steps):
            new_routes.append({"vehicle_id": r["vehicle_id"], "steps": new_steps})
    new_solution = {"routes": new_routes}
    # try to reinsert removed orders greedily using insert_order_greedy-like strategy: try candidate warehouses and vehicles
    for oid, sku_map in removed_items.items():
        inserted = False
        # find candidate warehouses
        for wh_id, orders in shipments_by_wh.items():
            if oid not in orders:
                continue
            wh_obj = env.warehouses[wh_id]
            # try to add as small single-order route on any vehicle
            for v in env.get_all_vehicles():
                # skip vehicles already used in new_solution
                used_vehicle_ids = {r["vehicle_id"] for r in new_solution.get("routes", [])}
                if v.id in used_vehicle_ids:
                    continue
                try:
                    home_node = env.get_vehicle_home_warehouse(v.id)
                except Exception:
                    home_node = getattr(wh_obj.location, "id", wh_obj.location)
                node_sequence = []
                # include warehouse pickup for the order's SKUs (always add; merge happens if at home)
                pickup_map = {sku: int(q) for sku, q in sku_map.items()}
                # ensure pickup entries include warehouse_id so validator applies warehouse inventory
                pickups_list = [{"warehouse_id": wh_id, "sku_id": sku, "quantity": int(q)} for sku, q in pickup_map.items()]
                node_sequence.append((getattr(wh_obj.location, "id", wh_obj.location), {"pickups": pickups_list}))
                node = env.get_order_location(oid)
                deliveries = [{"order_id": oid, "sku_id": sku, "quantity": int(q)} for sku, q in sku_map.items()]
                node_sequence.append((node, deliveries))
                steps = build_steps_with_path(env, node_sequence, home_node)
                if steps is None:
                    continue
                try:
                    valid, msg = env.validate_solution_complete({"routes": [{"vehicle_id": v.id, "steps": steps}]})
                except Exception:
                    valid, msg = True, "no-validate"
                if valid:
                    new_solution.setdefault("routes", []).append({"vehicle_id": v.id, "steps": steps})
                    inserted = True
                    break
            if inserted:
                break
    return new_solution


def memetic_solver(env: Any, time_budget: float = 15.0) -> Dict:
    start = time.time()
    # 1) allocate greedily
    shipments_by_wh = {}
    orders = env.get_all_order_ids()
    # naive allocation: pick first warehouse that can satisfy full order
    for oid in orders:
        req = env.get_order_requirements(oid)
        chosen = None
        for wid, wh in env.warehouses.items():
            inv = dict(env.get_warehouse_inventory(wid))
            if all(inv.get(sku, 0) >= q for sku, q in req.items()):
                chosen = wid
                break
        if chosen:
            shipments_by_wh.setdefault(chosen, {})[oid] = dict(req)
    # 2) create initial population
    pop = []
    pop_size = 1
    pop.append(greedy_initial_solution(env, shipments_by_wh))
    # random variants
    for _ in range(pop_size - 1):
        s = greedy_initial_solution(env, shipments_by_wh)
        # random removal of a few routes
        if s.get("routes"):
            for _ in range(random.randint(0, 2)):
                if not s["routes"]:
                    break
                s["routes"].pop(random.randrange(len(s["routes"])))
        pop.append(s)

    best = min(pop, key=lambda x: solution_cost(env, x))
    best_cost = solution_cost(env, best)

    # 3) evolution loop: crossover (route exchange), mutation (remove & reinsert), local search (mutate + accept)
    it = 0
    while time.time() - start < time_budget and it < 100:
        it += 1
        # pick parents
        a, b = random.sample(pop, 2)
        # simple crossover: take half routes from a and half from b
        routes_a = a.get("routes", [])
        routes_b = b.get("routes", [])
        child_routes = []
        for r in routes_a[:len(routes_a)//2]:
            child_routes.append(r)
        for r in routes_b[len(routes_b)//2:]:
            # avoid duplicate vehicles
            if any(cr["vehicle_id"] == r["vehicle_id"] for cr in child_routes):
                continue
            child_routes.append(r)
        # ensure unique vehicles in child
        seen = set()
        unique_routes = []
        for r in child_routes:
            if r["vehicle_id"] in seen:
                continue
            seen.add(r["vehicle_id"])
            unique_routes.append(r)
        child = {"routes": unique_routes}
        # mutation / local search
        child2 = mutate_solution(env, child, shipments_by_wh)
        # replace worst if better
        c_cost = solution_cost(env, child2)
        worst_idx = max(range(len(pop)), key=lambda i: solution_cost(env, pop[i]))
        if c_cost < solution_cost(env, pop[worst_idx]):
            pop[worst_idx] = child2
            if c_cost < best_cost:
                best = child2
                best_cost = c_cost
    # final local improvement: try to reinsert any missing orders greedily
    final = best
    return final


# entrypoint expected by environment

def solver(env: Any) -> Dict:
    return memetic_solver(env, time_budget=12.0)
