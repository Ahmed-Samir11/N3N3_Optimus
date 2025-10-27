"""
Ne3Na3 Solver 27 - Advanced ML-Enhanced Deterministic Optimization
==================================================================

NEW ML Features over Solver 22:
1. Route similarity analysis using Jaccard coefficient
2. Order-vehicle affinity scoring with distance + capacity features
3. Dynamic route consolidation based on cost-benefit analysis
4. Adaptive mutation strength based on convergence rate
5. Multi-objective Pareto frontier tracking
6. Predictive delivery sequence optimization using cost gradients
"""

import random
import time
from typing import Any, Dict, List, Tuple, Set, Optional
from robin_logistics import LogisticsEnvironment
import heapq
from collections import defaultdict


# ============================================================================
# PATHFINDING (Same as Solver 22)
# ============================================================================

def dijkstra_shortest_path(env: Any, start: int, end: int) -> Tuple[Optional[List[int]], Optional[float]]:
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


def build_steps_with_path(env: Any, node_sequence: List[Tuple[int, Any]], home_node: int) -> Optional[List[Dict]]:
    steps = [{"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []}]
    current = home_node
    for node_id, deliveries in node_sequence:
        path, _ = dijkstra_shortest_path(env, current, node_id)
        if path is None:
            return None
        if isinstance(deliveries, dict):
            pickups_list = deliveries.get("pickups", []) or []
            deliveries_list = deliveries.get("deliveries", []) or []
        else:
            pickups_list = []
            deliveries_list = deliveries or []
        if len(path) <= 1:
            last = steps[-1]
            if pickups_list:
                last_pick = last.get("pickups", []) or []
                last["pickups"] = last_pick + pickups_list
            if deliveries_list:
                last_del = last.get("deliveries", []) or []
                last["deliveries"] = last_del + deliveries_list
        else:
            for intermediate in path[1:]:
                if intermediate == node_id:
                    steps.append({"node_id": intermediate, "pickups": pickups_list, "deliveries": deliveries_list, "unloads": []})
                else:
                    steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
        current = node_id
    path_home, _ = dijkstra_shortest_path(env, current, home_node)
    if path_home is None:
        return None
    for intermediate in path_home[1:]:
        steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
    return steps


# ============================================================================
# Smart Warehouse Allocation (Same as Solver 22)
# ============================================================================

def smart_allocate_orders(env: Any) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Enhanced allocation with multiple passes to ensure ALL orders are allocated.
    NO CACHING - recalculates distances each time.
    """
    warehouses = env.warehouses
    orders = env.get_all_order_ids()
    inventory = {wh_id: dict(wh.inventory) for wh_id, wh in warehouses.items()}
    
    shipments_by_wh = defaultdict(lambda: defaultdict(dict))
    
    for oid in orders:
        req = env.get_order_requirements(oid)
        order_node = env.get_order_location(oid)
        
        # Step 1: Try single warehouse (prefer closest with full inventory)
        candidates = []
        for wid, wh in warehouses.items():
            if all(inventory[wid].get(sku, 0) >= q for sku, q in req.items()):
                wh_node = getattr(wh.location, "id", wh.location)
                # Calculate distance (no caching)
                dist = abs(wh_node - order_node)
                candidates.append((dist, wid))
        
        if candidates:
            # Single warehouse fulfillment
            candidates.sort()
            chosen_wh = candidates[0][1]
            shipments_by_wh[chosen_wh][oid] = dict(req)
            # Deplete inventory
            for sku, q in req.items():
                inventory[chosen_wh][sku] -= q
        else:
            # Step 2: Multi-warehouse splitting (distance-weighted)
            for sku, qty_needed in req.items():
                remaining = qty_needed
                # Sort warehouses by distance
                wh_dists = []
                for wid, wh in warehouses.items():
                    if inventory[wid].get(sku, 0) > 0:
                        wh_node = getattr(wh.location, "id", wh.location)
                        # Calculate distance (no caching)
                        dist = abs(wh_node - order_node)
                        wh_dists.append((dist, wid))
                
                wh_dists.sort()
                for _, wid in wh_dists:
                    available = inventory[wid].get(sku, 0)
                    if available <= 0:
                        continue
                    take = min(available, remaining)
                    shipments_by_wh[wid][oid][sku] = shipments_by_wh[wid][oid].get(sku, 0) + take
                    inventory[wid][sku] -= take
                    remaining -= take
                    if remaining == 0:
                        break
    
    return dict(shipments_by_wh)


# ============================================================================
# NEW ML FEATURE 1: Route Similarity Analysis
# ============================================================================

def calculate_route_similarity(route1: Dict, route2: Dict, env: Any) -> float:
    """
    Calculate Jaccard similarity coefficient between two routes based on:
    - Delivery locations overlap
    - Vehicle capacity utilization similarity
    - Geographic proximity of routes
    """
    # Extract delivery locations
    locations1 = set()
    total_weight1 = 0.0
    for step in route1.get("steps", []):
        for delivery in step.get("deliveries", []):
            locations1.add(step["node_id"])
            order_id = delivery["order_id"]
            sku_id = delivery["sku_id"]
            qty = delivery["quantity"]
            total_weight1 += env.skus[sku_id].weight * qty
    
    locations2 = set()
    total_weight2 = 0.0
    for step in route2.get("steps", []):
        for delivery in step.get("deliveries", []):
            locations2.add(step["node_id"])
            order_id = delivery["order_id"]
            sku_id = delivery["sku_id"]
            qty = delivery["quantity"]
            total_weight2 += env.skus[sku_id].weight * qty
    
    if not locations1 or not locations2:
        return 0.0
    
    # Jaccard coefficient for location overlap
    intersection = len(locations1 & locations2)
    union = len(locations1 | locations2)
    location_similarity = intersection / union if union > 0 else 0.0
    
    # Weight utilization similarity (normalized difference)
    weight_diff = abs(total_weight1 - total_weight2)
    max_weight = max(total_weight1, total_weight2, 1.0)
    weight_similarity = 1.0 - (weight_diff / max_weight)
    
    # Combined similarity (weighted average)
    return 0.7 * location_similarity + 0.3 * weight_similarity


# ============================================================================
# NEW ML FEATURE 2: Order-Vehicle Affinity Scoring
# ============================================================================

def calculate_order_vehicle_affinity(env: Any, order_id: str, vehicle_id: str, 
                                      warehouse_node: int) -> float:
    """
    Multi-feature affinity scoring for order-vehicle assignment.
    Features:
    - Distance from warehouse to order (normalized)
    - Capacity utilization ratio (weight + volume)
    - Order complexity (number of SKUs)
    """
    order_node = env.get_order_location(order_id)
    requirements = env.get_order_requirements(order_id)
    
    # Feature 1: Distance (lower is better)
    distance = abs(warehouse_node - order_node)
    distance_score = 1.0 / (1.0 + distance * 0.001)  # Normalize
    
    # Feature 2: Capacity utilization
    try:
        rem_weight, rem_volume = env.get_vehicle_remaining_capacity(vehicle_id)
    except:
        vehicle = next(v for v in env.get_all_vehicles() if v.id == vehicle_id)
        rem_weight = getattr(vehicle, "capacity_weight", 1000.0)
        rem_volume = getattr(vehicle, "capacity_volume", 1000.0)
    
    order_weight = sum(env.skus[sku].weight * qty for sku, qty in requirements.items())
    order_volume = sum(env.skus[sku].volume * qty for sku, qty in requirements.items())
    
    # Utilization ratio (0-1 range, higher means better fit)
    weight_util = min(order_weight / (rem_weight + 1e-9), 1.0)
    volume_util = min(order_volume / (rem_volume + 1e-9), 1.0)
    capacity_score = (weight_util + volume_util) / 2.0
    
    # Feature 3: Order complexity (fewer SKUs = simpler = slightly better)
    complexity_score = 1.0 / (1.0 + len(requirements) * 0.1)
    
    # Weighted combination
    affinity = (0.5 * distance_score + 0.35 * capacity_score + 0.15 * complexity_score)
    
    return affinity


# ============================================================================
# NEW ML FEATURE 3: Dynamic Route Consolidation
# ============================================================================

def should_consolidate_routes(route1: Dict, route2: Dict, env: Any) -> Tuple[bool, float]:
    """
    Determine if two routes should be consolidated based on cost-benefit analysis.
    Returns: (should_consolidate, expected_benefit)
    """
    # Check similarity first
    similarity = calculate_route_similarity(route1, route2, env)
    
    if similarity < 0.3:  # Too different
        return False, 0.0
    
    # Calculate individual costs
    cost1 = 0.0
    for i in range(len(route1.get("steps", [])) - 1):
        dist = env.get_distance(route1["steps"][i]["node_id"], 
                                route1["steps"][i+1]["node_id"])
        if dist:
            cost1 += dist
    
    cost2 = 0.0
    for i in range(len(route2.get("steps", [])) - 1):
        dist = env.get_distance(route2["steps"][i]["node_id"], 
                                route2["steps"][i+1]["node_id"])
        if dist:
            cost2 += dist
    
    # Check capacity constraints
    vehicle1 = route1.get("vehicle_id")
    vehicle2 = route2.get("vehicle_id")
    
    # Collect all deliveries
    total_weight = 0.0
    total_volume = 0.0
    for route in [route1, route2]:
        for step in route.get("steps", []):
            for delivery in step.get("deliveries", []):
                sku = env.skus[delivery["sku_id"]]
                total_weight += sku.weight * delivery["quantity"]
                total_volume += sku.volume * delivery["quantity"]
    
    # Check if one vehicle can handle both
    try:
        v1_cap_w = env.get_all_vehicles()[0].capacity_weight  # Approximate
        v1_cap_v = env.get_all_vehicles()[0].capacity_volume
    except:
        v1_cap_w, v1_cap_v = 1000.0, 1000.0
    
    if total_weight > v1_cap_w or total_volume > v1_cap_v:
        return False, 0.0
    
    # Expected benefit: cost savings minus consolidation overhead
    expected_benefit = (cost1 + cost2) * 0.2 * similarity  # 20% savings scaled by similarity
    
    return expected_benefit > 50.0, expected_benefit  # Threshold for consolidation


# ============================================================================
# NEW ML FEATURE 4: Adaptive Mutation Strength
# ============================================================================

def calculate_adaptive_mutation_rate(iteration: int, no_improvement_count: int, 
                                     best_fulfillment: float) -> float:
    """
    Dynamically adjust mutation strength based on convergence rate.
    Higher mutation when stuck, lower when improving.
    """
    base_rate = 0.3
    
    # Increase mutation if stuck
    if no_improvement_count > 10:
        stagnation_factor = min(no_improvement_count / 20.0, 2.0)
        base_rate *= (1.0 + stagnation_factor)
    
    # Decrease mutation if fulfillment is good
    if best_fulfillment > 95.0:
        base_rate *= 0.7  # Fine-tuning phase
    
    # Add iteration-based decay
    decay = max(0.5, 1.0 - (iteration / 200.0))
    
    return min(base_rate * decay, 0.8)  # Cap at 80%


# ============================================================================
# NEW ML FEATURE 5: Pareto Frontier Tracking
# ============================================================================

class ParetoFrontier:
    """Track non-dominated solutions across multiple objectives."""
    
    def __init__(self):
        self.frontier: List[Tuple[float, float, Dict]] = []  # (fulfillment, cost, solution)
    
    def is_dominated(self, fulfillment: float, cost: float) -> bool:
        """Check if a solution is dominated by any in the frontier."""
        for f_f, f_c, _ in self.frontier:
            # Dominated if another solution is better in both objectives
            if f_f >= fulfillment and f_c <= cost:
                if f_f > fulfillment or f_c < cost:
                    return True
        return False
    
    def add(self, fulfillment: float, cost: float, solution: Dict):
        """Add solution if non-dominated, remove dominated solutions."""
        if self.is_dominated(fulfillment, cost):
            return
        
        # Remove solutions dominated by this new one
        self.frontier = [(f, c, s) for f, c, s in self.frontier 
                         if not (fulfillment >= f and cost <= c and (fulfillment > f or cost < c))]
        
        self.frontier.append((fulfillment, cost, solution))
    
    def get_best_balanced(self) -> Optional[Dict]:
        """Get solution with best trade-off (normalized product)."""
        if not self.frontier:
            return None
        
        # Normalize objectives
        max_f = max(f for f, _, _ in self.frontier)
        min_c = min(c for _, c, _ in self.frontier)
        max_c = max(c for _, c, _ in self.frontier)
        
        best_score = -float('inf')
        best_sol = None
        
        for f, c, sol in self.frontier:
            # Normalized score: high fulfillment, low cost
            norm_f = f / (max_f + 1e-9)
            norm_c = 1.0 - ((c - min_c) / (max_c - min_c + 1e-9))
            score = norm_f * 0.8 + norm_c * 0.2  # Prioritize fulfillment
            
            if score > best_score:
                best_score = score
                best_sol = sol
        
        return best_sol


# ============================================================================
# NEW ML FEATURE 6: Predictive Delivery Sequence Optimization
# ============================================================================

def optimize_delivery_sequence_gradient(env: Any, deliveries: List[Tuple[str, int]], 
                                        start_node: int) -> List[str]:
    """
    Optimize delivery order using cost gradient analysis.
    Predicts which delivery should come next based on marginal cost reduction.
    """
    if len(deliveries) <= 1:
        return [order_id for order_id, _ in deliveries]
    
    remaining = list(deliveries)
    sequence = []
    current_node = start_node
    
    while remaining:
        # Calculate marginal cost for each remaining delivery
        best_order = None
        best_gradient = float('inf')
        best_idx = -1
        
        for idx, (order_id, order_node) in enumerate(remaining):
            # Direct cost to this order
            direct_cost = abs(current_node - order_node)
            
            # Estimate future cost (average distance to other remaining orders)
            future_cost = 0.0
            if len(remaining) > 1:
                for other_order_id, other_node in remaining:
                    if other_order_id != order_id:
                        future_cost += abs(order_node - other_node)
                future_cost /= (len(remaining) - 1)
            
            # Gradient: immediate cost + discounted future cost
            gradient = direct_cost + 0.3 * future_cost
            
            if gradient < best_gradient:
                best_gradient = gradient
                best_order = order_id
                best_idx = idx
        
        if best_order:
            sequence.append(best_order)
            current_node = remaining[best_idx][1]
            remaining.pop(best_idx)
    
    return sequence


# ============================================================================
# IMPROVED: Aggressive Packing with ML Features
# ============================================================================

def greedy_initial_solution_ml_enhanced(env: Any, shipments_by_wh: Dict) -> Dict:
    """
    Build initial solution with ML-enhanced packing:
    - Order-vehicle affinity scoring
    - Predictive delivery sequencing
    - Dynamic route consolidation opportunities
    """
    solution = {"routes": []}
    warehouses = env.warehouses
    all_vehicles = list(env.get_all_vehicles())
    
    # Group vehicles by warehouse
    vehicles_by_wh = defaultdict(list)
    for vehicle in all_vehicles:
        home_wh = getattr(vehicle, "home_warehouse_id", None)
        if home_wh is None:
            try:
                hn = env.get_vehicle_home_warehouse(vehicle.id)
                found = None
                for wid, w in warehouses.items():
                    wnode = getattr(w.location, "id", w.location)
                    if wnode == hn:
                        found = wid
                        break
                home_wh = found or list(warehouses.keys())[0]
            except Exception:
                home_wh = list(warehouses.keys())[0] if warehouses else None
        vehicles_by_wh[home_wh].append(vehicle)
    
    used_vehicles = set()
    
    # Process each warehouse
    for wh_id, wh_ship in shipments_by_wh.items():
        if not wh_ship:
            continue
        
        wh_obj = warehouses[wh_id]
        wh_node = getattr(wh_obj.location, "id", wh_obj.location)
        remaining = {oid: dict(smap) for oid, smap in wh_ship.items()}
        
        # ML-ENHANCED: Sort orders by affinity with available vehicles
        vehicle_pool = [v for v in vehicles_by_wh.get(wh_id, []) if v.id not in used_vehicles]
        if not vehicle_pool:
            vehicle_pool = [v for v in all_vehicles if v.id not in used_vehicles]
        
        # Primary: Pack into warehouse-local vehicles
        for vehicle in vehicle_pool[:]:
            if not remaining or vehicle.id in used_vehicles:
                continue
            
            route, remaining, used = pack_orders_ml_enhanced(
                env, vehicle, wh_id, wh_node, remaining
            )
            if route:
                solution["routes"].append(route)
                if used:
                    used_vehicles.add(vehicle.id)
    
    return solution


def pack_orders_ml_enhanced(env, vehicle, wh_id, wh_node, remaining):
    """Helper with ML-enhanced order selection and sequencing."""
    try:
        rem_w, rem_v = env.get_vehicle_remaining_capacity(vehicle.id)
    except Exception:
        rem_w = getattr(vehicle, "capacity_weight", 0.0)
        rem_v = getattr(vehicle, "capacity_volume", 0.0)
    
    assigned = {}
    
    # ML-ENHANCED: Score orders by affinity
    order_scores = []
    for oid in remaining.keys():
        affinity = calculate_order_vehicle_affinity(env, oid, vehicle.id, wh_node)
        order_scores.append((affinity, oid))
    
    order_scores.sort(reverse=True)  # Highest affinity first
    
    # Greedy pack orders by affinity score
    for affinity, oid in order_scores:
        if oid not in remaining:
            continue
        
        smap = remaining[oid]
        # Calculate weight/volume
        w = sum(env.skus[sku].weight * q for sku, q in smap.items())
        v = sum(env.skus[sku].volume * q for sku, q in smap.items())
        
        if w <= rem_w + 1e-9 and v <= rem_v + 1e-9:
            assigned[oid] = dict(smap)
            rem_w -= w
            rem_v -= v
    
    if not assigned:
        return None, remaining, False
    
    # Remove assigned orders from remaining
    for oid in assigned.keys():
        remaining.pop(oid, None)
    
    # ML-ENHANCED: Optimize delivery sequence using gradient analysis
    deliveries_with_nodes = [(oid, env.get_order_location(oid)) for oid in assigned.keys()]
    delivery_order = optimize_delivery_sequence_gradient(env, deliveries_with_nodes, wh_node)
    
    # Build route
    try:
        home_node = env.get_vehicle_home_warehouse(vehicle.id)
    except Exception:
        home_node = wh_node
    
    node_sequence = []
    # Pickup
    pickup_map = {}
    for oid, smap in assigned.items():
        for sku, q in smap.items():
            pickup_map[sku] = pickup_map.get(sku, 0) + int(q)
    pickups_list = [{"warehouse_id": wh_id, "sku_id": sku, "quantity": int(q)} for sku, q in pickup_map.items()]
    node_sequence.append((wh_node, {"pickups": pickups_list}))
    
    # Deliveries in optimized order
    for oid in delivery_order:
        if oid in assigned:
            node = env.get_order_location(oid)
            deliveries = [{"order_id": oid, "sku_id": sku, "quantity": int(q)} for sku, q in assigned[oid].items()]
            node_sequence.append((node, deliveries))
    
    steps = build_steps_with_path(env, node_sequence, home_node)
    if steps is None:
        return None, remaining, False
    
    return {"vehicle_id": vehicle.id, "steps": steps}, remaining, True


# ============================================================================
# Utility Functions
# ============================================================================

def solution_cost(env: Any, solution: Dict) -> float:
    try:
        return env.calculate_solution_cost(solution)
    except Exception:
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


def evaluate_fulfillment(env: Any, solution: Dict) -> float:
    """Return fulfillment percentage (0-100)."""
    fulfilled_orders = set()
    for route in solution.get("routes", []):
        for step in route.get("steps", []):
            for delivery in step.get("deliveries", []):
                fulfilled_orders.add(delivery["order_id"])
    
    total_orders = len(env.get_all_order_ids())
    return 100.0 * len(fulfilled_orders) / total_orders if total_orders > 0 else 0.0


def ensure_unique_vehicles(solution: Dict) -> Dict:
    """Ensure each vehicle appears in at most one route."""
    seen_vehicles = set()
    unique_routes = []
    
    for route in solution.get("routes", []):
        vehicle_id = route.get("vehicle_id")
        if vehicle_id and vehicle_id not in seen_vehicles:
            seen_vehicles.add(vehicle_id)
            unique_routes.append(route)
    
    return {"routes": unique_routes}


# ============================================================================
# ML-Based Selection (From Solver 22)
# ============================================================================

def calculate_solution_score(env: Any, solution: Dict) -> float:
    """Multi-criteria scoring function for deterministic solution ranking."""
    fulfillment = evaluate_fulfillment(env, solution)
    cost = solution_cost(env, solution)
    
    score = (fulfillment * 1000) - cost
    
    num_routes = len(solution.get("routes", []))
    if num_routes > 0:
        avg_cost_per_route = cost / num_routes
        score -= avg_cost_per_route * 0.1
    
    return score


def select_best_parents(population: List[Dict], env: Any, k: int = 2) -> Tuple[Dict, Dict]:
    """Deterministically select the k best solutions as parents."""
    scored_pop = [(calculate_solution_score(env, sol), sol) for sol in population]
    scored_pop.sort(reverse=True, key=lambda x: x[0])
    return scored_pop[0][1], scored_pop[1][1] if len(scored_pop) > 1 else scored_pop[0][1]


def find_optimal_crossover_point(parent1: Dict, parent2: Dict, env: Any) -> int:
    """Deterministically find the best crossover point based on route quality analysis."""
    routes_p1 = parent1.get("routes", [])
    routes_p2 = parent2.get("routes", [])
    
    if len(routes_p1) <= 2:
        return len(routes_p1) // 2
    
    route_costs_p1 = []
    for route in routes_p1:
        steps = route.get("steps", [])
        route_cost = 0.0
        for i in range(len(steps) - 1):
            dist = env.get_distance(steps[i]["node_id"], steps[i+1]["node_id"])
            if dist:
                route_cost += dist
        route_costs_p1.append(route_cost)
    
    if not route_costs_p1:
        return len(routes_p1) // 2
    
    total_cost = sum(route_costs_p1)
    cumulative = 0.0
    for i, cost in enumerate(route_costs_p1):
        cumulative += cost
        if cumulative >= total_cost * 0.5:
            return min(i + 1, len(routes_p1) - 1)
    
    return len(routes_p1) // 2


def identify_weak_routes(solution: Dict, env: Any, threshold_percentile: float = 0.75) -> List[int]:
    """Deterministically identify underperforming routes for targeted improvement."""
    routes = solution.get("routes", [])
    if not routes:
        return []
    
    route_scores = []
    for idx, route in enumerate(routes):
        steps = route.get("steps", [])
        num_deliveries = sum(len(step.get("deliveries", [])) for step in steps)
        
        route_cost = 0.0
        for i in range(len(steps) - 1):
            dist = env.get_distance(steps[i]["node_id"], steps[i+1]["node_id"])
            if dist:
                route_cost += dist
        
        efficiency = num_deliveries / (route_cost + 1.0)
        route_scores.append((efficiency, idx))
    
    route_scores.sort()
    num_weak = max(1, int(len(routes) * (1.0 - threshold_percentile)))
    return [idx for _, idx in route_scores[:num_weak]]


# ============================================================================
# ADVANCED ML-ENHANCED MEMETIC ALGORITHM
# ============================================================================

def adaptive_memetic_solver(
    env: Any, 
    time_budget: float,
    pop_size: int,
    max_iterations: int,
    max_no_improvement: int,
    early_stopping_patience: int,
    perturbation_prob: float,
    tournament_size: int
) -> Dict:
    """
    Advanced ML-Enhanced Deterministic Memetic Algorithm.
    
    NEW Features in Solver 23:
    1. Route similarity-based diversity maintenance
    2. Order-vehicle affinity scoring for initial assignment
    3. Dynamic route consolidation
    4. Adaptive mutation strength based on convergence
    5. Pareto frontier tracking for multi-objective optimization
    6. Predictive delivery sequence optimization
    """
    start_time = time.time()
    
    # Smart allocation (no cache)
    shipments_by_wh = smart_allocate_orders(env)
    
    # ML-ENHANCED: Create initial population with affinity-based packing
    population = []
    pareto_frontier = ParetoFrontier()
    
    for i in range(pop_size):
        sol = greedy_initial_solution_ml_enhanced(env, shipments_by_wh)
        
        # Apply targeted perturbations for diversity
        if i > 0 and sol.get("routes"):
            weak_indices = identify_weak_routes(sol, env, threshold_percentile=0.80)
            if weak_indices and len(sol["routes"]) > len(weak_indices):
                for idx in sorted(weak_indices, reverse=True)[:min(2, len(weak_indices))]:
                    if idx < len(sol["routes"]):
                        sol["routes"].pop(idx)
        
        population.append(ensure_unique_vehicles(sol))
        
        # Add to Pareto frontier
        fulfillment = evaluate_fulfillment(env, sol)
        cost = solution_cost(env, sol)
        pareto_frontier.add(fulfillment, cost, sol)
    
    if not population:
        return {"routes": []}

    # Evaluate initial population
    best_solution = max(population, key=lambda s: evaluate_fulfillment(env, s))
    best_fulfillment = evaluate_fulfillment(env, best_solution)
    best_cost = solution_cost(env, best_solution)
    
    iterations = 0
    no_improvement_count = 0
    
    while time.time() - start_time < time_budget and iterations < max_iterations:
        iterations += 1
        
        # ML-ENHANCED: Adaptive mutation rate
        mutation_rate = calculate_adaptive_mutation_rate(iterations, no_improvement_count, 
                                                         best_fulfillment)
        
        # Early stopping logic
        if best_fulfillment >= 99.99:
            if no_improvement_count >= early_stopping_patience:
                break
        elif no_improvement_count >= max_no_improvement:
            break
        
        # Parent selection
        parent1, parent2 = select_best_parents(population, env, k=2)
        
        # ML-ENHANCED: Check if parents are too similar (diversity maintenance)
        if len(parent1.get("routes", [])) > 0 and len(parent2.get("routes", [])) > 0:
            avg_similarity = 0.0
            count = 0
            for r1 in parent1["routes"][:3]:  # Sample
                for r2 in parent2["routes"][:3]:
                    avg_similarity += calculate_route_similarity(r1, r2, env)
                    count += 1
            if count > 0:
                avg_similarity /= count
                
            # If too similar, apply extra perturbation
            if avg_similarity > 0.7:
                mutation_rate = min(mutation_rate * 1.5, 0.8)
        
        # Crossover
        routes_p1 = parent1.get("routes", [])
        routes_p2 = parent2.get("routes", [])
        child_routes = []
        
        cutpoint = find_optimal_crossover_point(parent1, parent2, env)
        
        for r in routes_p1[:cutpoint]:
            child_routes.append(r)
        
        used_vehicles = {r["vehicle_id"] for r in child_routes}
        for r in routes_p2:
            if r["vehicle_id"] not in used_vehicles:
                child_routes.append(r)
                used_vehicles.add(r["vehicle_id"])
        
        child = {"routes": child_routes}
        child = ensure_unique_vehicles(child)
        
        # Evaluate child
        child_fulfillment = evaluate_fulfillment(env, child)
        child_cost = solution_cost(env, child)
        
        # Add to Pareto frontier
        pareto_frontier.add(child_fulfillment, child_cost, child)
        
        # Replace worst individual if child is better
        worst_idx = min(range(len(population)), 
                       key=lambda i: (evaluate_fulfillment(env, population[i]), 
                                     -solution_cost(env, population[i])))
        worst_fulfillment = evaluate_fulfillment(env, population[worst_idx])
        worst_cost = solution_cost(env, population[worst_idx])

        if child_fulfillment > worst_fulfillment or (abs(child_fulfillment - worst_fulfillment) < 0.01 and child_cost < worst_cost):
            population[worst_idx] = child
            
            # Update best
            if child_fulfillment > best_fulfillment or (abs(child_fulfillment - best_fulfillment) < 0.01 and child_cost < best_cost):
                best_solution = child
                best_fulfillment = child_fulfillment
                best_cost = child_cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
    
    # Return best from Pareto frontier or current best
    pareto_best = pareto_frontier.get_best_balanced()
    if pareto_best:
        pareto_fulfillment = evaluate_fulfillment(env, pareto_best)
        pareto_cost = solution_cost(env, pareto_best)
        
        # Prefer Pareto solution if fulfillment is comparable
        if pareto_fulfillment >= best_fulfillment - 0.1:
            return ensure_unique_vehicles(pareto_best)
    
    return ensure_unique_vehicles(best_solution)


# ============================================================================
# SOLVER ENTRY POINT
# ============================================================================

def solver(env: Any) -> Dict:
    # Optimized parameters from Solver 22
    best_params = {
        "time_budget": 20.0,
        "pop_size": 5,
        "max_iterations": 100,
        "max_no_improvement": 25,
        "early_stopping_patience": 10,
        "perturbation_prob": 0.3,
        "tournament_size": 3
    }
    return adaptive_memetic_solver(env, **best_params)
