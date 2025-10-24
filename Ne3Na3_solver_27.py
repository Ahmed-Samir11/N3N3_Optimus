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
        """Return neighbor nodes from adjacency list (keys may be strings)."""
        # Try both string and int keys to handle different adjacency list formats
        if str(node) in self.adj:
            return self.adj[str(node)]
        if node in self.adj:
            return self.adj[node]
        if int(node) in self.adj:
            return self.adj[int(node)]
        return []

    def _dijkstra_path(self, start: int, goal: int) -> Optional[List[int]]:
        """Dijkstra's shortest path algorithm with path reconstruction."""
        if start == goal:
            return [start]
        
        import heapq
        
        pq = [(0, start, [start])]
        visited = set()
        
        while pq:
            dist, current, path = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                return path
            
            neighbors = self._neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (dist + 1, neighbor, new_path))
        
        return None

    def shortest_path(self, start: int, goal: int) -> Optional[List[int]]:
        """Shortest path using Dijkstra's algorithm."""
        return self._dijkstra_path(start, goal)

    @staticmethod
    def path_distance(path: Optional[List[int]]) -> float:
        """Edge count distance if no weights given (fallback)."""
        if not path or len(path) < 2:
            return 0.0
        return float(len(path) - 1)

    def _route_distance(self, vehicle_id: str, warehouse_node: int, order_node: int) -> float:
        """Calculate total route distance Home→Warehouse→Order→Home."""
        home_node = int(self.env.get_vehicle_home_warehouse(vehicle_id))
        
        p1 = self.shortest_path(home_node, warehouse_node)
        p2 = self.shortest_path(warehouse_node, order_node)
        p3 = self.shortest_path(order_node, home_node)
        
        if p1 is None or p2 is None or p3 is None:
            return float("inf")
        
        return self.path_distance(p1) + self.path_distance(p2) + self.path_distance(p3)
    
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
    def _best_wh_cost_for(self, order_id: str, vehicle_id: str, alpha: float = 0.5) -> Tuple[float, float, Optional[str]]:
        """Return (min_combined_score, distance, best_warehouse_id) for an order-vehicle pair.
        
        Args:
            alpha: weight for cost vs distance (0=only distance, 1=only cost, 0.5=balanced)
        
        Combined score = alpha * normalized_cost + (1-alpha) * normalized_distance
        """
        if not self.check_capacity(order_id, vehicle_id):
            return float("inf"), float("inf"), None
        o_node = int(self.env.get_order_location(order_id))
        
        # First pass: collect all feasible options to normalize
        options: List[Tuple[str, float, float]] = []  # (warehouse_id, cost, distance)
        for w in (self.env.warehouses or {}).keys():
            if not self.can_fulfill_from(order_id, w):
                continue
            w_node = int(getattr(self.env.warehouses[w].location, "id", self.env.warehouses[w].location))
            c = self.cost_heuristic(vehicle_id, w_node, o_node)
            if c == float("inf"):
                continue
            d = self._route_distance(vehicle_id, w_node, o_node)
            if d == float("inf"):
                continue
            options.append((w, c, d))
        
        if not options:
            return float("inf"), float("inf"), None
        
        # Normalize cost and distance
        min_cost = min(opt[1] for opt in options)
        max_cost = max(opt[1] for opt in options)
        min_dist = min(opt[2] for opt in options)
        max_dist = max(opt[2] for opt in options)
        
        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
        dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
        
        best_score = float("inf")
        best_wh = None
        best_dist = float("inf")
        
        for w, c, d in options:
            norm_cost = (c - min_cost) / cost_range
            norm_dist = (d - min_dist) / dist_range
            score = alpha * norm_cost + (1.0 - alpha) * norm_dist
            if score < best_score:
                best_score, best_wh, best_dist = score, w, d
        
        return best_score, best_dist, best_wh
    
    def _route_distance(self, vehicle_id: str, warehouse_node: int, order_node: int) -> float:
        """Calculate total route distance Home→Warehouse→Order→Home."""
        home_node = int(self.env.get_vehicle_home_warehouse(vehicle_id))
                
        p1 = self.shortest_path(home_node, warehouse_node)
        p2 = self.shortest_path(warehouse_node, order_node)
        p3 = self.shortest_path(order_node, home_node)
        
        if p1 is None or p2 is None or p3 is None:
            return float("inf")
        
        d1 = self.path_distance(p1)
        d2 = self.path_distance(p2)
        d3 = self.path_distance(p3)
        total = d1 + d2 + d3
        
        return total

    def _build_cost_matrix(self, alpha: float = 0.5) -> Tuple[List[List[float]], List[str], List[str], List[List[Optional[str]]]]:
        """Build cost matrix with configurable cost/distance balance.
        
        Args:
            alpha: 0.0 = minimize distance only, 1.0 = minimize cost only, 0.5 = balanced
        """
        orders: List[str] = self.env.get_all_order_ids() or []
        vehicles: List[str] = self.env.get_available_vehicles() or []
        cost: List[List[float]] = []
        best_wh: List[List[Optional[str]]] = []
        for o in orders:
            row: List[float] = []
            row_wh: List[Optional[str]] = []
            for v in vehicles:
                score, _, w = self._best_wh_cost_for(o, v, alpha)
                row.append(score if score != float("inf") else self.BIG)
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

    def assign_orders_hungarian(self, alpha: float = 0.5) -> List[Tuple[str, str, str]]:
        """Assign orders to vehicles with Hungarian on balanced cost/distance score.
        
        Args:
            alpha: 0.0 = prioritize short distance, 1.0 = prioritize low cost, 0.5 = balanced
        
        Returns list of (order_id, vehicle_id, warehouse_id).
        """
        cost, orders, vehicles, best_wh = self._build_cost_matrix(alpha)
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

    def assign_orders_greedy_consolidated(self, alpha: float = 0.7) -> List[Tuple[str, str, str]]:
        """Greedy assignment that allows multiple orders per vehicle for better consolidation.
        
        Args:
            alpha: 0.0 = prioritize distance, 1.0 = prioritize cost, 0.7 = cost priority
        
        Returns list of (order_id, vehicle_id, warehouse_id).
        """
        orders: List[str] = self.env.get_all_order_ids() or []
        vehicles: List[str] = self.env.get_available_vehicles() or []
        wh_ids: List[str] = list((self.env.warehouses or {}).keys())
        
        # Track cumulative weight/volume per vehicle
        vehicle_load: Dict[str, Tuple[float, float]] = {v: (0.0, 0.0) for v in vehicles}
        vehicle_warehouse: Dict[str, Optional[str]] = {v: None for v in vehicles}
        
        result: List[Tuple[str, str, str]] = []
        
        # Sort orders by weight (descending) to pack heavy items first
        order_weights = [(o, self.order_weight_volume(o)[0]) for o in orders]
        order_weights.sort(key=lambda x: x[1], reverse=True)
        sorted_orders = [o for o, _ in order_weights]
        
        for o in sorted_orders:
            o_node = int(self.env.get_order_location(o))
            order_w, order_v = self.order_weight_volume(o)
            
            best: Optional[Tuple[float, str, str]] = None  # (score, vehicle, warehouse)
            
            for v in vehicles:
                vehicle = self.env.get_vehicle_by_id(v)
                cap_w = float(getattr(vehicle, "capacity_weight", float("inf")))
                cap_v = float(getattr(vehicle, "capacity_volume", float("inf")))
                
                # Check if adding this order exceeds capacity
                curr_w, curr_v = vehicle_load[v]
                if curr_w + order_w > cap_w or curr_v + order_v > cap_v:
                    continue
                
                # If vehicle already assigned to a warehouse, prefer same warehouse
                preferred_wh = vehicle_warehouse[v]
                
                for w in wh_ids:
                    if not self.can_fulfill_from(o, w):
                        continue
                    
                    # If vehicle already has a warehouse, only consider that one
                    if preferred_wh is not None and w != preferred_wh:
                        continue
                    
                    w_node = int(getattr(self.env.warehouses[w].location, "id", 
                                       self.env.warehouses[w].location))
                    
                    # Calculate score based on alpha
                    c = self.cost_heuristic(v, w_node, o_node)
                    if c == float("inf"):
                        continue
                    
                    d = self._route_distance(v, w_node, o_node)
                    if d == float("inf"):
                        continue
                    
                    # Bonus for reusing same vehicle (saves fixed cost)
                    reuse_bonus = -500.0 if curr_w > 0 else 0.0  # Incentivize consolidation
                    
                    score = alpha * c + (1.0 - alpha) * d + reuse_bonus
                    
                    if (best is None) or (score < best[0]):
                        best = (score, v, w)
            
            if best is not None:
                _, v_sel, w_sel = best
                result.append((o, v_sel, w_sel))
                
                # Update vehicle load
                curr_w, curr_v = vehicle_load[v_sel]
                vehicle_load[v_sel] = (curr_w + order_w, curr_v + order_v)
                vehicle_warehouse[v_sel] = w_sel
        
        return result


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
                    # CRITICAL FIX: Only merge if same vehicle AND same warehouse
                    # This prevents assigning one vehicle to multiple routes
                    if v1 != v2 or w1 != w2:
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

    def _two_opt_order_sequence(self, vehicle_id: str, warehouse_id: str, order_ids: List[str]) -> List[str]:
        """Apply 2-opt to improve order visit sequence. Returns optimized order list."""
        if len(order_ids) <= 2:
            return order_ids
        
        home = int(self.env.get_vehicle_home_warehouse(vehicle_id))
        wnode = int(getattr(self.env.warehouses[warehouse_id].location, "id", self.env.warehouses[warehouse_id].location))
        order_nodes = [int(self.env.get_order_location(o)) for o in order_ids]
        
        def route_length(seq_indices: List[int]) -> float:
            """Calculate total distance for a given order sequence."""
            dist = self._dist(home, wnode)
            if dist == float('inf'):
                return dist
            prev = wnode
            for idx in seq_indices:
                d = self._dist(prev, order_nodes[idx])
                if d == float('inf'):
                    return d
                dist += d
                prev = order_nodes[idx]
            d = self._dist(prev, home)
            if d == float('inf'):
                return float('inf')
            return dist + d
        
        # Start with current sequence
        current_seq = list(range(len(order_ids)))
        current_dist = route_length(current_seq)
        
        improved = True
        max_iterations = 50  # Limit iterations to keep runtime reasonable
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(len(current_seq) - 1):
                for j in range(i + 2, len(current_seq)):
                    # Try reversing segment [i+1:j+1]
                    new_seq = current_seq[:i+1] + current_seq[i+1:j+1][::-1] + current_seq[j+1:]
                    new_dist = route_length(new_seq)
                    if new_dist < current_dist:
                        current_seq = new_seq
                        current_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
        
        # Return reordered order_ids
        return [order_ids[i] for i in current_seq]

    def build_multi_order_route_optimized(self, vehicle_id: str, warehouse_id: str, order_ids: List[str]) -> Optional[Dict]:
        """Build multi-order route with 2-opt optimization."""
        if not order_ids:
            return None
        
        # Apply 2-opt to find better order sequence
        optimized_orders = self._two_opt_order_sequence(vehicle_id, warehouse_id, order_ids)
        
        # Build route with optimized sequence
        home = int(self.env.get_vehicle_home_warehouse(vehicle_id))
        wnode = int(getattr(self.env.warehouses[warehouse_id].location, "id", self.env.warehouses[warehouse_id].location))
        order_nodes = [int(self.env.get_order_location(o)) for o in optimized_orders]
        
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
        combined = self._combined_requirements(optimized_orders)
        steps[wi]["pickups"] = [
            {"warehouse_id": warehouse_id, "sku_id": sku, "quantity": int(qty)} for sku, qty in combined.items()
        ]
        # deliveries: at each order location
        for oid in optimized_orders:
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
    
    # Add these methods to the SimpleOptimizer class, after build_multi_order_route_optimized

    # -------------------------
    # Multi-warehouse fallback
    # -------------------------
    def assign_orders_multi_wh_fallback(self, assigned_orders: List[str]) -> List[Tuple[str, str, List[Tuple[str, Dict[str, int]]]]]:
        """Fallback for unassigned orders: allow multi-warehouse pickups.
        
        Returns list of (order_id, vehicle_id, plan) where plan is:
        [(warehouse_id, {sku: qty}), ...]
        """
        all_orders = set(self.env.get_all_order_ids() or [])
        unassigned = all_orders - set(assigned_orders)
        
        result: List[Tuple[str, str, List[Tuple[str, Dict[str, int]]]]] = []
        vehicles: List[str] = self.env.get_available_vehicles() or []
        
        for order_id in unassigned:
            reqs = self.env.get_order_requirements(order_id) or {}
            if not reqs:
                continue
            
            # Try to find a vehicle that can carry the total weight/volume
            for vehicle_id in vehicles:
                if not self.check_capacity(order_id, vehicle_id):
                    continue
                
                # Plan multi-warehouse pickups to fulfill all SKUs
                plan = self._plan_multi_wh_pickups(order_id, reqs)
                if plan:
                    result.append((order_id, vehicle_id, plan))
                    break
        
        return result
    
    def _plan_multi_wh_pickups(self, order_id: str, requirements: Dict[str, int]) -> Optional[List[Tuple[str, Dict[str, int]]]]:
        """Plan which warehouses to visit to fulfill requirements.
        
        Returns list of (warehouse_id, {sku: qty_to_pick}) or None if impossible.
        """
        remaining = dict(requirements)
        plan: List[Tuple[str, Dict[str, int]]] = []
        
        for warehouse_id in (self.env.warehouses or {}).keys():
            if not remaining:
                break
            
            inv = self.env.get_warehouse_inventory(warehouse_id) or {}
            pickups: Dict[str, int] = {}
            
            for sku_id in list(remaining.keys()):
                needed = remaining[sku_id]
                available = inv.get(sku_id, 0)
                take = min(needed, available)
                
                if take > 0:
                    pickups[sku_id] = take
                    remaining[sku_id] -= take
                    if remaining[sku_id] == 0:
                        del remaining[sku_id]
            
            if pickups:
                plan.append((warehouse_id, pickups))
        
        # Check if we fulfilled everything
        if remaining:
            return None  # Can't fulfill completely
        
        return plan if plan else None
    
    def build_route_multi_wh(self, order_id: str, vehicle_id: str, plan: List[Tuple[str, Dict[str, int]]]) -> Optional[Dict]:
        """Build route with multiple warehouse pickups.
        
        Args:
            plan: [(warehouse_id, {sku: qty}), ...]
        """
        if not plan:
            return None
        
        home = int(self.env.get_vehicle_home_warehouse(vehicle_id))
        order_node = int(self.env.get_order_location(order_id))
        
        # Build path: home -> wh1 -> wh2 -> ... -> order -> home
        warehouse_nodes = []
        for wh_id, _ in plan:
            wh_node = int(getattr(self.env.warehouses[wh_id].location, "id", 
                                 self.env.warehouses[wh_id].location))
            warehouse_nodes.append((wh_id, wh_node))
        
        # Build segments
        segments: List[List[int]] = []
        
        # Home to first warehouse
        p = self.shortest_path(home, warehouse_nodes[0][1])
        if p is None:
            return None
        segments.append(p)
        
        # Between warehouses
        for i in range(len(warehouse_nodes) - 1):
            p = self.shortest_path(warehouse_nodes[i][1], warehouse_nodes[i+1][1])
            if p is None:
                return None
            segments.append(p[1:])  # Skip duplicate node
        
        # Last warehouse to order
        p = self.shortest_path(warehouse_nodes[-1][1], order_node)
        if p is None:
            return None
        segments.append(p[1:])
        
        # Order back to home
        p = self.shortest_path(order_node, home)
        if p is None:
            return None
        segments.append(p[1:])
        
        # Flatten path
        full_path: List[int] = []
        for seg in segments:
            full_path.extend(seg)
        
        # Build steps
        steps: List[Dict] = []
        for node in full_path:
            steps.append({"node_id": int(node), "pickups": [], "deliveries": [], "unloads": []})
        
        # Add pickups at each warehouse
        for wh_id, wh_node in warehouse_nodes:
            try:
                wi = full_path.index(wh_node)
            except ValueError:
                return None
            
            # Find what to pick from this warehouse from the plan
            for plan_wh_id, pickups in plan:
                if plan_wh_id == wh_id:
                    for sku_id, qty in pickups.items():
                        steps[wi]["pickups"].append({
                            "warehouse_id": wh_id,
                            "sku_id": sku_id,
                            "quantity": int(qty)
                        })
                    break
        
        # Add delivery at order location
        try:
            oi = full_path.index(order_node)
        except ValueError:
            return None
        
        full_reqs = self.env.get_order_requirements(order_id) or {}
        for sku_id, qty in full_reqs.items():
            steps[oi]["deliveries"].append({
                "order_id": order_id,
                "sku_id": sku_id,
                "quantity": int(qty)
            })
        
        return {"vehicle_id": vehicle_id, "steps": steps}


def solver(env: LogisticsEnvironment, alpha: float = 0.7, use_2opt: bool = True) -> Dict:
    """Main entry: Greedy consolidation + 2-opt optimization."""
    sol: Dict = {"routes": []}
    opt = SimpleOptimizer(env)
    used_vehicles: set = set()

    # Use greedy consolidation
    assignments = opt.assign_orders_greedy_consolidated(alpha=alpha)
    if not assignments:
        return sol

    # Group orders by (vehicle, warehouse)
    from collections import defaultdict
    groups_dict: dict = defaultdict(list)
    for (o, v, w) in assignments:
        groups_dict[(v, w)].append(o)
    
    groups = [(v, w, orders) for (v, w), orders in groups_dict.items()]

    # Build routes
    for vehicle_id, warehouse_id, order_ids in groups:
        if vehicle_id in used_vehicles:
            continue
        
        if len(order_ids) == 1:
            route = opt.build_route(order_ids[0], vehicle_id, warehouse_id)
        else:
            if use_2opt:
                route = opt.build_multi_order_route_optimized(vehicle_id, warehouse_id, order_ids)
            else:
                route = opt.build_multi_order_route(vehicle_id, warehouse_id, order_ids)
        
        if route is not None:
            sol["routes"].append(route)
            used_vehicles.add(vehicle_id)

    # Multi-warehouse fallback
    assigned_orders = {o for (_, _, os) in groups for o in os}
    mw_assignments = opt.assign_orders_multi_wh_fallback(list(assigned_orders))
    
    for (order_id, vehicle_id, plan) in mw_assignments:
        if vehicle_id in used_vehicles:
            continue
        route = opt.build_route_multi_wh(order_id, vehicle_id, plan)
        if route is not None:
            sol["routes"].append(route)
            used_vehicles.add(vehicle_id)
    
    return sol


def my_solver(env: LogisticsEnvironment) -> Dict:
    """Wrapper with optimized defaults: prioritize cost efficiency."""
    return solver(env, alpha=0.7, use_2opt=True)


if __name__ == "__main__":
    env = LogisticsEnvironment()
    solution = solver(env)
    print(f"✅ Created {len(solution.get('routes', []))} routes")
    
    # Check if routes have proper steps
    if solution.get('routes'):
        first_route = solution['routes'][0]
        print(f"\nFirst route has {len(first_route.get('steps', []))} steps")
        if first_route.get('steps'):
            print(f"First 3 steps: {[s['node_id'] for s in first_route['steps'][:3]]}")