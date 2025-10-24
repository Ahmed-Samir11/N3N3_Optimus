#!/usr/bin/env python3
"""
Hackathon Solver — Cost-aware assignment + BFS routing

This solver assigns each order to a feasible vehicle and warehouse,
then builds a simple Home → Warehouse → Customer → Home route using
shortest paths on the provided road network.

It is designed to produce a valid solution quickly and clearly, and can
be extended with savings/2-opt if needed.
"""

from typing import Dict, List, Optional, Tuple

# The environment is provided by the hackathon package
from robin_logistics import LogisticsEnvironment


class SimpleOptimizer:
    """Encapsulates assignment, pathfinding, and route building."""

    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        rn = env.get_road_network_data() or {}
        self.adj = rn.get("adjacency_list", {})
        self.BIG = 1e12  # large cost for infeasible entries

    # -------------------------
    # Utilities
    # -------------------------
    def _neighbors(self, node: int) -> List[int]:
        """Return neighbor nodes from adjacency list (supports str/int keys and dict/list forms)."""
        neighbors = None
        # Try string key first
        key_str = str(node)
        if isinstance(self.adj, dict) and key_str in self.adj:
            neighbors = self.adj[key_str]
        # Fallback: try integer key
        if neighbors is None and isinstance(self.adj, dict) and node in self.adj:
            neighbors = self.adj[node]
        if neighbors is None:
            return []
        # If dict of neighbor->meta
        if isinstance(neighbors, dict):
            return [int(k) for k in neighbors.keys()]
        # If list/tuple of neighbor ids
        if isinstance(neighbors, (list, tuple)):
            return [int(x) for x in neighbors]
        return []

    def _bfs_path(self, start: int, goal: int) -> Optional[List[int]]:
        if start == goal:
            return [start]
        from collections import deque
        visited = set([start])
        q = deque([(start, [start])])
        while q:
            cur, path = q.popleft()
            for nei in self._neighbors(cur):
                if nei in visited:
                    continue
                if nei == goal:
                    return path + [nei]
                visited.add(nei)
                q.append((nei, path + [nei]))
        return None

    def shortest_path(self, start: int, goal: int) -> Optional[List[int]]:
        """Shortest path (unweighted BFS) without caching."""
        return self._bfs_path(start, goal)

    @staticmethod
    def path_distance(path: Optional[List[int]]) -> float:
        """Edge count distance if no weights given (fallback)."""
        if not path or len(path) < 2:
            return 0.0
        return float(len(path) - 1)

    # -------------------------
    # Feasibility checks
    # -------------------------
    def order_weight_volume(self, order_id: str) -> Tuple[float, float]:
        reqs = self.env.get_order_requirements(order_id) or {}
        # Assume env.skus is a dict sku_id -> object with weight/volume
        skus = getattr(self.env, "skus", {})
        # Note: expected SKU ids include 'Light_Item', 'Medium_Item', 'Heavy_Item'.
        # We use whatever weights/volumes the environment provides for these ids.
        total_w = 0.0
        total_v = 0.0
        for sku_id, qty in reqs.items():
            sku = skus.get(sku_id)
            w = float(getattr(sku, "weight", 0.0)) if sku else 0.0
            v = float(getattr(sku, "volume", 0.0)) if sku else 0.0
            total_w += w * float(qty)
            total_v += v * float(qty)
        return total_w, total_v

    def check_capacity(self, order_id: str, vehicle_id: str) -> bool:
        vw, vv = self.order_weight_volume(order_id)
        vehicle = self.env.get_vehicle_by_id(vehicle_id)
        cap_w = float(getattr(vehicle, "capacity_weight", float("inf")))
        cap_v = float(getattr(vehicle, "capacity_volume", float("inf")))
        return (vw <= cap_w) and (vv <= cap_v)

    def can_fulfill_from(self, order_id: str, warehouse_id: str) -> bool:
        reqs = self.env.get_order_requirements(order_id) or {}
        inv = self.env.get_warehouse_inventory(warehouse_id) or {}
        for sku_id, qty in reqs.items():
            if inv.get(sku_id, 0) < qty:
                return False
        return True

    # -------------------------
    # Assignment
    # -------------------------
    def _best_wh_cost_for(self, order_id: str, vehicle_id: str) -> Tuple[float, Optional[str]]:
        """Return (min_cost, best_warehouse_id) for an order-vehicle pair, or (inf, None)."""
        if not self.check_capacity(order_id, vehicle_id):
            return float("inf"), None
        o_node = int(self.env.get_order_location(order_id))
        best_cost = float("inf")
        best_wh = None
        for w in (self.env.warehouses or {}).keys():
            if not self.can_fulfill_from(order_id, w):
                continue
            w_node = int(getattr(self.env.warehouses[w].location, "id", self.env.warehouses[w].location))
            c = self.cost_heuristic(vehicle_id, w_node, o_node)
            if c < best_cost:
                best_cost, best_wh = c, w
        return best_cost, best_wh

    def _build_cost_matrix(self) -> Tuple[List[List[float]], List[str], List[str], List[List[Optional[str]]]]:
        orders: List[str] = self.env.get_all_order_ids() or []
        vehicles: List[str] = self.env.get_available_vehicles() or []
        cost: List[List[float]] = []
        best_wh: List[List[Optional[str]]] = []
        for o in orders:
            row: List[float] = []
            row_wh: List[Optional[str]] = []
            for v in vehicles:
                c, w = self._best_wh_cost_for(o, v)
                row.append(c if c != float("inf") else self.BIG)
                row_wh.append(w)
            cost.append(row)
            best_wh.append(row_wh)
        return cost, orders, vehicles, best_wh

    @staticmethod
    def _pad_to_square(cost: List[List[float]], pad_val: float) -> Tuple[List[List[float]], int, int]:
        """Pad rectangular matrix with pad_val to make it square; return (padded, n_rows, n_cols)."""
        if not cost:
            return [], 0, 0
        r = len(cost)
        c = max(len(row) for row in cost)
        n = max(r, c)
        # normalize rows to same length first
        norm = [row + [pad_val] * (c - len(row)) for row in cost]
        if r < n:
            for _ in range(n - r):
                norm.append([pad_val] * c)
        # now pad columns to n
        if c < n:
            norm = [row + [pad_val] * (n - c) for row in norm]
        return norm, r, c

    def _hungarian(self, cost: List[List[float]]) -> List[Tuple[int, int]]:
        """Classic O(n^3) Hungarian algorithm for square matrices.
        Returns list of (row_idx, col_idx) assignments.
        """
        n = len(cost)
        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(0, n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        assignment = [0] * (n + 1)
        for j in range(1, n + 1):
            assignment[p[j]] = j
        result = []
        for i in range(1, n + 1):
            result.append((i - 1, assignment[i] - 1))
        return result

    def assign_orders_hungarian(self) -> List[Tuple[str, str, str]]:
        """Assign orders to vehicles with Hungarian on min cost; choose best warehouse per pair.
        Returns list of (order_id, vehicle_id, warehouse_id).
        """
        cost, orders, vehicles, best_wh = self._build_cost_matrix()
        if not orders or not vehicles:
            return []
        # Pad to square
        padded, r, c = self._pad_to_square(cost, self.BIG)
        n = max(r, c)
        if n == 0:
            return []
        pairs = self._hungarian(padded)
        assignments: List[Tuple[str, str, str]] = []
        for (ri, cj) in pairs:
            if ri < r and cj < c:
                if cost[ri][cj] >= self.BIG:
                    continue  # infeasible
                o_id = orders[ri]
                v_id = vehicles[cj]
                w_id = best_wh[ri][cj]
                if w_id is None:
                    continue
                # capacity & inventory already checked when building costs
                assignments.append((o_id, v_id, w_id))
        return assignments
    def cost_heuristic(self, vehicle_id: str, warehouse_node: int, order_node: int) -> float:
        vehicle = self.env.get_vehicle_by_id(vehicle_id)
        home_node = int(self.env.get_vehicle_home_warehouse(vehicle_id))

        p1 = self.shortest_path(home_node, warehouse_node)
        p2 = self.shortest_path(warehouse_node, order_node)
        p3 = self.shortest_path(order_node, home_node)
        if p1 is None or p2 is None or p3 is None:
            return float("inf")
        dist = self.path_distance(p1) + self.path_distance(p2) + self.path_distance(p3)

        fixed = float(getattr(vehicle, "fixed_cost", 0.0))
        var = float(getattr(vehicle, "cost_per_km", 0.0))
        maxd = float(getattr(vehicle, "max_distance", float("inf")))
        if dist > maxd:
            return float("inf")
        return fixed + var * dist

    def assign_orders(self) -> List[Tuple[str, str, str]]:
        """Greedy, cost-aware: returns list of (order_id, vehicle_id, warehouse_id)."""
        orders: List[str] = self.env.get_all_order_ids() or []
        vehicles: List[str] = self.env.get_available_vehicles() or []
        wh_ids: List[str] = list((self.env.warehouses or {}).keys())

        used: set = set()
        result: List[Tuple[str, str, str]] = []

        for o in orders:
            o_node = int(self.env.get_order_location(o))

            best: Optional[Tuple[float, str, str]] = None  # (cost, v, w)
            for v in vehicles:
                if v in used:
                    continue
                if not self.check_capacity(o, v):
                    continue
                for w in wh_ids:
                    if not self.can_fulfill_from(o, w):
                        continue
                    w_node = int(getattr(self.env.warehouses[w].location, "id", self.env.warehouses[w].location))
                    c = self.cost_heuristic(v, w_node, o_node)
                    if c == float("inf"):
                        continue
                    if (best is None) or (c < best[0]):
                        best = (c, v, w)

            if best is not None:
                _, v_sel, w_sel = best
                used.add(v_sel)
                result.append((o, v_sel, w_sel))

        return result

    # -------------------------
    # Multi-warehouse fulfillment (single vehicle)
    # -------------------------
    def _plan_multi_wh_pickups(self, order_id: str) -> List[Tuple[str, Dict[str, int]]]:
        """Greedy plan to fulfill an order from multiple warehouses.
        Returns a list of (warehouse_id, {sku: qty_to_pick}) covering the order, or [] if impossible.
        """
        reqs = dict(self.env.get_order_requirements(order_id) or {})
        remaining = {sku: int(q) for sku, q in reqs.items()}
        plan: List[Tuple[str, Dict[str, int]]] = []
        if not remaining:
            return plan
        wh_ids: List[str] = list((self.env.warehouses or {}).keys())
        # Simple greedy: iterate warehouses and take what they can supply
        for w in wh_ids:
            inv = self.env.get_warehouse_inventory(w) or {}
            take: Dict[str, int] = {}
            any_take = False
            for sku, need in list(remaining.items()):
                if need <= 0:
                    continue
                avail = int(inv.get(sku, 0))
                if avail <= 0:
                    continue
                qty = min(need, avail)
                if qty > 0:
                    take[sku] = qty
                    remaining[sku] -= qty
                    any_take = True
            if any_take:
                plan.append((w, take))
            # Check if done
            if all(qty <= 0 for qty in remaining.values()):
                break
        # If still remaining, impossible
        if any(qty > 0 for qty in remaining.values()):
            return []
        return plan

    def _estimate_multi_wh_cost(self, vehicle_id: str, order_node: int, wh_nodes: List[int]) -> float:
        vehicle = self.env.get_vehicle_by_id(vehicle_id)
        home = int(self.env.get_vehicle_home_warehouse(vehicle_id))
        # Route: home -> w1 -> w2 -> ... -> order -> home
        total = 0.0
        prev = home
        for wn in wh_nodes:
            p = self.shortest_path(prev, wn)
            if p is None:
                return float('inf')
            total += self.path_distance(p)
            prev = wn
        p = self.shortest_path(prev, order_node)
        if p is None:
            return float('inf')
        total += self.path_distance(p)
        p = self.shortest_path(order_node, home)
        if p is None:
            return float('inf')
        total += self.path_distance(p)
        fixed = float(getattr(vehicle, "fixed_cost", 0.0))
        var = float(getattr(vehicle, "cost_per_km", 0.0))
        maxd = float(getattr(vehicle, "max_distance", float('inf')))
        if total > maxd:
            return float('inf')
        return fixed + var * total

    def assign_orders_multi_wh_fallback(self, already_assigned: List[str]) -> List[Tuple[str, str, List[Tuple[str, Dict[str, int]]]]]:
        """Assign remaining orders allowing multi-warehouse pickups on a single vehicle.
        Returns list of (order_id, vehicle_id, pickups_plan).
        """
        assigned_orders = set(already_assigned)
        orders: List[str] = self.env.get_all_order_ids() or []
        vehicles: List[str] = self.env.get_available_vehicles() or []
        results: List[Tuple[str, str, List[Tuple[str, Dict[str, int]]]]] = []
        for o in orders:
            if o in assigned_orders:
                continue
            o_node = int(self.env.get_order_location(o))
            best = None  # (cost, vehicle_id, plan)
            plan = self._plan_multi_wh_pickups(o)
            if not plan:
                continue
            # capacity check for total load
            vw, vv = self.order_weight_volume(o)
            for v in vehicles:
                vehicle = self.env.get_vehicle_by_id(v)
                cap_w = float(getattr(vehicle, "capacity_weight", float('inf')))
                cap_v = float(getattr(vehicle, "capacity_volume", float('inf')))
                if vw > cap_w or vv > cap_v:
                    continue
                wh_nodes = [int(getattr(self.env.warehouses[w].location, "id", self.env.warehouses[w].location)) for (w, _) in plan]
                c = self._estimate_multi_wh_cost(v, o_node, wh_nodes)
                if c == float('inf'):
                    continue
                if best is None or c < best[0]:
                    best = (c, v, plan)
            if best is not None:
                _, v_sel, plan_sel = best
                results.append((o, v_sel, plan_sel))
        return results

    # -------------------------
    # Route building
    # -------------------------
    def build_route(self, order_id: str, vehicle_id: str, warehouse_id: str) -> Optional[Dict]:
        """Return a route dict with expanded node steps and proper actions."""
        reqs = self.env.get_order_requirements(order_id) or {}

        home_node = int(self.env.get_vehicle_home_warehouse(vehicle_id))
        w_node = int(getattr(self.env.warehouses[warehouse_id].location, "id", self.env.warehouses[warehouse_id].location))
        o_node = int(self.env.get_order_location(order_id))

        p1 = self.shortest_path(home_node, w_node)
        p2 = self.shortest_path(w_node, o_node)
        p3 = self.shortest_path(o_node, home_node)
        if p1 is None or p2 is None or p3 is None:
            return None

        # Merge paths, avoiding duplicate node at joins
        full_path: List[int] = []
        for seg in (p1, p2[1:], p3[1:]):
            full_path.extend(seg)

        # Build steps: empty actions on transit nodes, pickups at warehouse, deliveries at customer
        steps: List[Dict] = []
        for idx, node in enumerate(full_path):
            step = {"node_id": int(node), "pickups": [], "deliveries": [], "unloads": []}
            steps.append(step)

        # Add pickup actions at first occurrence of warehouse node
        try:
            w_index = full_path.index(w_node)
            pickups = []
            for sku_id, qty in reqs.items():
                pickups.append({"warehouse_id": warehouse_id, "sku_id": sku_id, "quantity": int(qty)})
            steps[w_index]["pickups"] = pickups
        except ValueError:
            return None

        # Add delivery actions at first occurrence of order node
        try:
            o_index = full_path.index(o_node)
            deliveries = []
            for sku_id, qty in reqs.items():
                deliveries.append({"order_id": order_id, "sku_id": sku_id, "quantity": int(qty)})
            steps[o_index]["deliveries"] = deliveries
        except ValueError:
            return None

        return {"vehicle_id": vehicle_id, "steps": steps}

    # -------------------------
    # Clarke–Wright merging (pairwise)
    # -------------------------
    def _combined_requirements(self, order_ids: List[str]) -> Dict[str, int]:
        combined: Dict[str, int] = {}
        for oid in order_ids:
            for sku, qty in (self.env.get_order_requirements(oid) or {}).items():
                combined[sku] = combined.get(sku, 0) + int(qty)
        return combined

    def _can_fulfill_combined(self, order_ids: List[str], warehouse_id: str) -> bool:
        reqs = self._combined_requirements(order_ids)
        inv = self.env.get_warehouse_inventory(warehouse_id) or {}
        return all(inv.get(s, 0) >= q for s, q in reqs.items())

    def _check_capacity_combined(self, order_ids: List[str], vehicle_id: str) -> bool:
        # sum weights/volumes
        skus = getattr(self.env, "skus", {})
        total_w = 0.0
        total_v = 0.0
        for oid in order_ids:
            for sku_id, qty in (self.env.get_order_requirements(oid) or {}).items():
                sku = skus.get(sku_id)
                w = float(getattr(sku, "weight", 0.0)) if sku else 0.0
                v = float(getattr(sku, "volume", 0.0)) if sku else 0.0
                total_w += w * float(qty)
                total_v += v * float(qty)
        vehicle = self.env.get_vehicle_by_id(vehicle_id)
        cap_w = float(getattr(vehicle, "capacity_weight", float("inf")))
        cap_v = float(getattr(vehicle, "capacity_volume", float("inf")))
        return (total_w <= cap_w) and (total_v <= cap_v)

    def _dist(self, a: int, b: int) -> float:
        p = self.shortest_path(a, b)
        if p is None:
            return float('inf')
        return self.path_distance(p)

    def clarke_wright_merge(self, assignments: List[Tuple[str, str, str]]) -> List[Tuple[str, str, List[str]]]:
        """Pairwise merge orders onto a vehicle when beneficial.
        Input: list of (order, vehicle, warehouse)
        Output: list of (vehicle, warehouse, [orders]) possibly merged.
        """
        if not assignments:
            return []
        # Start with each as its own route group
        groups: List[Tuple[str, str, List[str]]] = [(v, w, [o]) for (o, v, w) in assignments]
        changed = True
        while changed:
            changed = False
            best_gain = 0.0
            best_pair = None  # (i, j, new_group)
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    v1, w1, os1 = groups[i]
                    v2, w2, os2 = groups[j]
                    # Only merge if same warehouse to keep inventory logic simple
                    if w1 != w2:
                        continue
                    # Try using vehicle v1 to serve os1 + os2
                    order_nodes = [int(self.env.get_order_location(o)) for o in os1 + os2]
                    home = int(self.env.get_vehicle_home_warehouse(v1))
                    wnode = int(getattr(self.env.warehouses[w1].location, "id", self.env.warehouses[w1].location))
                    # Savings formula on first two orders only (simple heuristic)
                    if len(order_nodes) < 2:
                        continue
                    i_node, j_node = order_nodes[0], order_nodes[1]
                    s = self._dist(home, i_node) + self._dist(home, j_node) - self._dist(i_node, j_node)
                    if s <= 0:
                        continue
                    # Capacity & inventory checks
                    cand_orders = os1 + os2
                    if not self._check_capacity_combined(cand_orders, v1):
                        continue
                    if not self._can_fulfill_combined(cand_orders, w1):
                        continue
                    # Estimate cost before and after
                    # Before: two separate tours via same warehouse
                    before = 0.0
                    for o in (os1 + os2):
                        on = int(self.env.get_order_location(o))
                        d = self._dist(home, wnode) + self._dist(wnode, on) + self._dist(on, home)
                        vehicle = self.env.get_vehicle_by_id(v1)
                        before += float(getattr(vehicle, "fixed_cost", 0.0)) + float(getattr(vehicle, "cost_per_km", 0.0)) * d
                    # After: merged sequence home->w->i->j->home cost
                    d_after = self._dist(home, wnode) + self._dist(wnode, i_node) + self._dist(i_node, j_node) + self._dist(j_node, home)
                    vehicle = self.env.get_vehicle_by_id(v1)
                    after = float(getattr(vehicle, "fixed_cost", 0.0)) + float(getattr(vehicle, "cost_per_km", 0.0)) * d_after
                    gain = before - after
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, j, (v1, w1, cand_orders))
            if best_pair is not None:
                i, j, newg = best_pair
                # Replace j with new merged group, remove i
                groups.pop(j)
                groups.pop(i)
                groups.append(newg)
                changed = True
        return groups

    def build_multi_order_route(self, vehicle_id: str, warehouse_id: str, order_ids: List[str]) -> Optional[Dict]:
        if not order_ids:
            return None
        # simple sequence: O1 then O2 then ...
        home = int(self.env.get_vehicle_home_warehouse(vehicle_id))
        wnode = int(getattr(self.env.warehouses[warehouse_id].location, "id", self.env.warehouses[warehouse_id].location))
        order_nodes = [int(self.env.get_order_location(o)) for o in order_ids]
        # Build path segments
        segments: List[List[int]] = []
        p = self.shortest_path(home, wnode)
        if p is None:
            return None
        segments.append(p)
        for idx, on in enumerate(order_nodes):
            prev = wnode if idx == 0 else order_nodes[idx - 1]
            sp = self.shortest_path(prev, on)
            if sp is None:
                return None
            segments.append(sp[1:] if idx > 0 else sp)
        # last back home
        last = order_nodes[-1]
        sp = self.shortest_path(last, home)
        if sp is None:
            return None
        segments.append(sp[1:])
        # Flatten
        full_path: List[int] = []
        for seg in segments:
            full_path.extend(seg)
        # Steps with actions
        steps: List[Dict] = []
        for node in full_path:
            steps.append({"node_id": int(node), "pickups": [], "deliveries": [], "unloads": []})
        # pickups: combined at warehouse
        try:
            wi = full_path.index(wnode)
        except ValueError:
            return None
        combined = self._combined_requirements(order_ids)
        steps[wi]["pickups"] = [
            {"warehouse_id": warehouse_id, "sku_id": sku, "quantity": int(qty)} for sku, qty in combined.items()
        ]
        # deliveries: at each order location
        for oid in order_ids:
            on = int(self.env.get_order_location(oid))
            try:
                oi = full_path.index(on)
            except ValueError:
                return None
            dels = []
            for sku, qty in (self.env.get_order_requirements(oid) or {}).items():
                dels.append({"order_id": oid, "sku_id": sku, "quantity": int(qty)})
            steps[oi]["deliveries"].extend(dels)
        return {"vehicle_id": vehicle_id, "steps": steps}

    def build_route_multi_wh(self, order_id: str, vehicle_id: str, plan: List[Tuple[str, Dict[str, int]]]) -> Optional[Dict]:
        """Build route with multiple warehouse pickups for a single order."""
        home = int(self.env.get_vehicle_home_warehouse(vehicle_id))
        o_node = int(self.env.get_order_location(order_id))
        wh_nodes: List[Tuple[str, int]] = [
            (w, int(getattr(self.env.warehouses[w].location, "id", self.env.warehouses[w].location))) for (w, _) in plan
        ]
        # Build path segments: home -> w1 -> w2 -> ... -> order -> home
        segments: List[List[int]] = []
        prev = home
        for _, wn in wh_nodes:
            sp = self.shortest_path(prev, wn)
            if sp is None:
                return None
            if segments:
                segments.append(sp[1:])
            else:
                segments.append(sp)
            prev = wn
        sp = self.shortest_path(prev, o_node)
        if sp is None:
            return None
        segments.append(sp[1:])
        sp = self.shortest_path(o_node, home)
        if sp is None:
            return None
        segments.append(sp[1:])
        # Flatten
        full_path: List[int] = []
        for seg in segments:
            full_path.extend(seg)
        # Steps with actions
        steps: List[Dict] = []
        for node in full_path:
            steps.append({"node_id": int(node), "pickups": [], "deliveries": [], "unloads": []})
        # pickups at each warehouse as per plan
        for (w, items) in plan:
            wnode = int(getattr(self.env.warehouses[w].location, "id", self.env.warehouses[w].location))
            try:
                wi = full_path.index(wnode)
            except ValueError:
                return None
            steps[wi]["pickups"].extend([
                {"warehouse_id": w, "sku_id": sku, "quantity": int(qty)} for sku, qty in items.items()
            ])
        # deliveries at order
        try:
            oi = full_path.index(o_node)
        except ValueError:
            return None
        for sku, qty in (self.env.get_order_requirements(order_id) or {}).items():
            steps[oi]["deliveries"].append({"order_id": order_id, "sku_id": sku, "quantity": int(qty)})
        return {"vehicle_id": vehicle_id, "steps": steps}


def solver(env: LogisticsEnvironment) -> Dict:
    """Main entry: Hungarian assignment + Clarke–Wright merging, with greedy fallback."""
    sol: Dict = {"routes": []}
    opt = SimpleOptimizer(env)

    # Try Hungarian first
    assignments = opt.assign_orders_hungarian()
    if not assignments:
        # Fallback to greedy
        assignments = opt.assign_orders()

    # Try to merge routes with Clarke–Wright on single-warehouse assignments
    groups = opt.clarke_wright_merge(assignments)
    if not groups:
        groups = [(v, w, [o]) for (o, v, w) in assignments]

    for (vehicle_id, warehouse_id, order_ids) in groups:
        if len(order_ids) == 1:
            route = opt.build_route(order_ids[0], vehicle_id, warehouse_id)
        else:
            route = opt.build_multi_order_route(vehicle_id, warehouse_id, order_ids)
        if route is not None:
            sol["routes"].append(route)

    # Fallback for orders left unassigned: allow multi-warehouse pickups on a single vehicle
    assigned_orders = {o for (_, _, os) in groups for o in os}
    mw_assignments = opt.assign_orders_multi_wh_fallback(list(assigned_orders))
    for (order_id, vehicle_id, plan) in mw_assignments:
        route = opt.build_route_multi_wh(order_id, vehicle_id, plan)
        if route is not None:
            sol["routes"].append(route)

    return sol


def my_solver(env: LogisticsEnvironment) -> Dict:
    """Wrapper used by the dashboard runner."""
    return solver(env)

# if __name__ == "__main__":
#     env = LogisticsEnvironment()
#     solution = my_solver(env)
#     print(f"Built {len(solution.get('routes', []))} routes.")