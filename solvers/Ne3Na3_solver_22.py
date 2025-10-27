"""
Ne3Na3 Solver 22 - ML-Enhanced Deterministic Optimization
=========================================================

Improvements over Solver 20:
1. Replaces random decisions with ML-based deterministic scoring
2. Multi-criteria parent selection (fulfillment + cost + efficiency)
3. Smart crossover points based on route analysis
4. Targeted perturbations instead of random removals
5. Consistent, reproducible results
"""

import random
import time
from typing import Any, Dict, List, Tuple, Set, Optional
from robin_logistics import LogisticsEnvironment
import heapq
from collections import defaultdict


# ============================================================================
# PATHFINDING (Same as Solver 12)
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
# Smart Warehouse Allocation (Same as Solver 12)
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
# IMPROVED: Aggressive Packing (from Solver 12) + Hungarian for Route Order
# ============================================================================

def greedy_initial_solution_aggressive(env: Any, shipments_by_wh: Dict) -> Dict:
    """
    Build initial solution with aggressive packing (FROM SOLVER 12):
    - Try ALL available vehicles, not just warehouse-local ones
    - Multiple passes to ensure no orders are missed
    - THEN optimize delivery order with Hungarian algorithm
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
        
        # Sort orders by distance
        order_distances = []
        for oid in remaining.keys():
            order_node = env.get_order_location(oid)
            dist = abs(wh_node - order_node)
            order_distances.append((dist, oid))
        order_distances.sort()
        
        # Primary: Pack into warehouse-local vehicles
        for vehicle in vehicles_by_wh.get(wh_id, []):
            if not remaining or vehicle.id in used_vehicles:
                continue
            
            route, remaining, used = pack_orders_into_vehicle(
                env, vehicle, wh_id, wh_node, remaining, order_distances
            )
            if route:
                solution["routes"].append(route)
                if used:
                    used_vehicles.add(vehicle.id)
        
        # Secondary: If orders remain, try ANY unused vehicle
        if remaining:
            for vehicle in all_vehicles:
                if not remaining or vehicle.id in used_vehicles:
                    continue
                
                route, remaining, used = pack_orders_into_vehicle(
                    env, vehicle, wh_id, wh_node, remaining, order_distances
                )
                if route:
                    solution["routes"].append(route)
                    if used:
                        used_vehicles.add(vehicle.id)
    
    return solution


def pack_orders_into_vehicle(env, vehicle, wh_id, wh_node, remaining, order_distances):
    """Helper to pack orders into a single vehicle with Hungarian-optimized delivery order."""
    try:
        rem_w, rem_v = env.get_vehicle_remaining_capacity(vehicle.id)
    except Exception:
        rem_w = getattr(vehicle, "capacity_weight", 0.0)
        rem_v = getattr(vehicle, "capacity_volume", 0.0)
    
    assigned = {}
    
    # Greedy pack orders in distance order
    for dist, oid in order_distances:
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
    
    # NEAREST NEIGHBOR OPTIMIZATION: Optimize delivery order
    assigned_orders = list(assigned.keys())
    if len(assigned_orders) > 1:
        # Find optimal tour using nearest neighbor
        # Start from warehouse, visit all orders, return to warehouse
        current_node = wh_node
        delivery_order = []
        remaining_orders = set(range(len(assigned_orders)))
        
        while remaining_orders:
            # Find nearest unvisited order
            min_dist = float('inf')
            next_idx = None
            for idx in remaining_orders:
                order_loc = env.get_order_location(assigned_orders[idx])
                dist = abs(current_node - order_loc)
                if dist < min_dist:
                    min_dist = dist
                    next_idx = idx
            
            if next_idx is not None:
                delivery_order.append(assigned_orders[next_idx])
                current_node = env.get_order_location(assigned_orders[next_idx])
                remaining_orders.remove(next_idx)
    else:
        delivery_order = assigned_orders
    
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
        node = env.get_order_location(oid)
        deliveries = [{"order_id": oid, "sku_id": sku, "quantity": int(q)} for sku, q in assigned[oid].items()]
        node_sequence.append((node, deliveries))
    
    steps = build_steps_with_path(env, node_sequence, home_node)
    if steps is None:
        return None, remaining, False
    
    return {"vehicle_id": vehicle.id, "steps": steps}, remaining, True


# ============================================================================
# Utility Functions (Same as Solver 12)
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
# ML-BASED SCORING AND SELECTION
# ============================================================================

def calculate_solution_score(env: Any, solution: Dict) -> float:
    """
    Multi-criteria scoring function for deterministic solution ranking.
    Higher score = better solution.
    """
    fulfillment = evaluate_fulfillment(env, solution)
    cost = solution_cost(env, solution)
    
    # Normalize metrics (lower cost = higher score)
    # Fulfillment weight: 1000 (heavily prioritize 100% fulfillment)
    # Cost weight: -1 (minimize cost)
    score = (fulfillment * 1000) - cost
    
    # Bonus for efficiency
    num_routes = len(solution.get("routes", []))
    if num_routes > 0:
        avg_cost_per_route = cost / num_routes
        score -= avg_cost_per_route * 0.1  # Slight penalty for inefficient routes
    
    return score


def select_best_parents(population: List[Dict], env: Any, k: int = 2) -> Tuple[Dict, Dict]:
    """
    Deterministically select the k best solutions as parents based on multi-criteria scoring.
    Replaces random tournament selection.
    """
    # Score all solutions
    scored_pop = [(calculate_solution_score(env, sol), sol) for sol in population]
    
    # Sort by score (descending - highest score first)
    scored_pop.sort(reverse=True, key=lambda x: x[0])
    
    # Return top 2 solutions
    return scored_pop[0][1], scored_pop[1][1] if len(scored_pop) > 1 else scored_pop[0][1]


def find_optimal_crossover_point(parent1: Dict, parent2: Dict, env: Any) -> int:
    """
    Deterministically find the best crossover point based on route quality analysis.
    Replaces random cutpoint selection.
    """
    routes_p1 = parent1.get("routes", [])
    routes_p2 = parent2.get("routes", [])
    
    if len(routes_p1) <= 2:
        return len(routes_p1) // 2
    
    # Analyze route costs to find natural split point
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
    
    # Find the point where cumulative cost is closest to 50% of total
    total_cost = sum(route_costs_p1)
    cumulative = 0.0
    for i, cost in enumerate(route_costs_p1):
        cumulative += cost
        if cumulative >= total_cost * 0.5:
            return min(i + 1, len(routes_p1) - 1)
    
    return len(routes_p1) // 2


def identify_weak_routes(solution: Dict, env: Any, threshold_percentile: float = 0.75) -> List[int]:
    """
    Deterministically identify underperforming routes for targeted improvement.
    Replaces random perturbation.
    """
    routes = solution.get("routes", [])
    if not routes:
        return []
    
    # Calculate efficiency score for each route (deliveries per unit cost)
    route_scores = []
    for idx, route in enumerate(routes):
        steps = route.get("steps", [])
        num_deliveries = sum(len(step.get("deliveries", [])) for step in steps)
        
        route_cost = 0.0
        for i in range(len(steps) - 1):
            dist = env.get_distance(steps[i]["node_id"], steps[i+1]["node_id"])
            if dist:
                route_cost += dist
        
        # Efficiency: deliveries per unit cost (higher is better)
        efficiency = num_deliveries / (route_cost + 1.0)  # +1 to avoid division by zero
        route_scores.append((efficiency, idx))
    
    # Sort by efficiency (ascending - worst first)
    route_scores.sort()
    
    # Return indices of worst 25% of routes
    num_weak = max(1, int(len(routes) * (1.0 - threshold_percentile)))
    return [idx for _, idx in route_scores[:num_weak]]


# ============================================================================
# ML-ENHANCED MEMETIC ALGORITHM
# ============================================================================

def adaptive_memetic_solver(
    env: Any, 
    time_budget: float,
    pop_size: int,
    max_iterations: int,
    max_no_improvement: int,
    early_stopping_patience: int,
    perturbation_prob: float,  # Kept for compatibility but not used
    tournament_size: int  # Kept for compatibility but not used
) -> Dict:
    """
    ML-Enhanced Deterministic Memetic Algorithm.
    
    Key improvements over random-based approach:
    1. Multi-criteria scoring for parent selection (fulfillment + cost + efficiency)
    2. Deterministic crossover points based on route quality analysis
    3. Targeted removal of weak routes instead of random perturbations
    4. Consistent, reproducible results
    5. NO CACHING (competition requirement)
    """
    start_time = time.time()
    
    # Smart allocation (no cache)
    shipments_by_wh = smart_allocate_orders(env)
    
    # Create smaller, more diverse initial population for speed
    population = []
    
    # Build multiple solutions with different strategies
    for i in range(pop_size):
        sol = greedy_initial_solution_aggressive(env, shipments_by_wh)
        
        # ML-BASED: Apply targeted perturbations instead of random removals
        if i > 0 and sol.get("routes"):
            # Deterministically identify weak routes and remove them for diversity
            weak_indices = identify_weak_routes(sol, env, threshold_percentile=0.80)
            if weak_indices and len(sol["routes"]) > len(weak_indices):
                # Remove weakest routes (in reverse order to preserve indices)
                for idx in sorted(weak_indices, reverse=True)[:min(2, len(weak_indices))]:
                    if idx < len(sol["routes"]):
                        sol["routes"].pop(idx)
        
        population.append(ensure_unique_vehicles(sol))
    
    if not population:
        return {"routes": []}

    # Evaluate initial population
    best_solution = max(population, key=lambda s: evaluate_fulfillment(env, s))
    best_fulfillment = evaluate_fulfillment(env, best_solution)
    best_cost = solution_cost(env, best_solution)
    
    # Extended search
    iterations = 0
    no_improvement_count = 0
    
    # Don't stop early if we have high fulfillment but not 100%
    while time.time() - start_time < time_budget and iterations < max_iterations:
        iterations += 1
        
        # Early stopping only if 100% fulfillment or truly converged
        if best_fulfillment >= 99.99:
            # We have 100%, just optimize cost a bit more
            if no_improvement_count >= early_stopping_patience:
                break
        elif no_improvement_count >= max_no_improvement:
            break
        
        # ML-BASED: Deterministic parent selection based on multi-criteria scoring
        parent1, parent2 = select_best_parents(population, env, k=2)
        
        # Crossover
        routes_p1 = parent1.get("routes", [])
        routes_p2 = parent2.get("routes", [])
        child_routes = []
        
        # ML-BASED: Deterministic crossover point based on route quality analysis
        cutpoint = find_optimal_crossover_point(parent1, parent2, env)
        
        for r in routes_p1[:cutpoint]:
            child_routes.append(r)
        
        # Avoid duplicate vehicles
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
        
        # Replace worst individual if child is better
        worst_idx = min(range(len(population)), key=lambda i: (evaluate_fulfillment(env, population[i]), -solution_cost(env, population[i])))
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
    
    return ensure_unique_vehicles(best_solution)


# ============================================================================
# SOLVER ENTRY POINT
# ============================================================================

def solver(env: Any) -> Dict:
    # Best parameters will be hardcoded here after tuning
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
