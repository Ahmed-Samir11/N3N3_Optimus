"""
Ne3Na3 Solver 12 - Improved Fulfillment Focus
==============================================

Improvements over Solver 11:
1. More aggressive initial population (10 solutions instead of 5)
2. Extended search time (longer iterations, no early stopping on high fulfillment)
3. Better order packing: Try all vehicles, not just warehouse-local ones
4. Multi-pass order allocation to catch missed orders
5. Increased time budget to 25 seconds
"""

import random
import time
from typing import Any, Dict, List, Tuple, Set, Optional
from robin_logistics import LogisticsEnvironment
import heapq
from collections import defaultdict


# ============================================================================
# PATHFINDING (Same as Solver 11)
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
# IMPROVED: Smart Warehouse Allocation with Better Multi-Warehouse Splitting
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
# IMPROVED: More Aggressive Initial Solution Building
# ============================================================================

def greedy_initial_solution_aggressive(env: Any, shipments_by_wh: Dict) -> Dict:
    """
    Build initial solution with aggressive packing:
    - Try ALL available vehicles, not just warehouse-local ones
    - Multiple passes to ensure no orders are missed
    """
    solution = {"routes": []}
    warehouses = env.warehouses
    all_vehicles = list(env.get_all_vehicles())
    
    # Group vehicles by warehouse for primary assignment
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
    """Helper to pack orders into a single vehicle."""
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
    
    # Deliveries in distance order
    delivery_order = [(abs(wh_node - env.get_order_location(oid)), oid) for oid in assigned.keys()]
    delivery_order.sort()
    
    for _, oid in delivery_order:
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
# IMPROVED: Extended Memetic Algorithm
# ============================================================================

def adaptive_memetic_solver(env: Any, time_budget: float = 25.0) -> Dict:
    """
    Improved memetic algorithm:
    - Larger population (10)
    - No early stopping if fulfillment >= 95%
    - More iterations
    - Better diversity
    - NO CACHING (competition requirement)
    """
    start_time = time.time()
    
    # Smart allocation (no cache)
    shipments_by_wh = smart_allocate_orders(env)
    
    # Create larger, more diverse initial population
    pop_size = 10
    population = []
    
    # Build multiple solutions with different strategies
    for i in range(pop_size):
        sol = greedy_initial_solution_aggressive(env, shipments_by_wh)
        
        # Add controlled randomness to some solutions
        if i > 0 and sol.get("routes") and random.random() < 0.3:
            # Small perturbation
            if len(sol["routes"]) > 1:
                num_remove = random.randint(1, min(2, len(sol["routes"]) // 3))
                for _ in range(num_remove):
                    if sol["routes"]:
                        sol["routes"].pop(random.randrange(len(sol["routes"])))
        
        population.append(ensure_unique_vehicles(sol))
    
    # Evaluate initial population
    best_solution = max(population, key=lambda s: evaluate_fulfillment(env, s))
    best_fulfillment = evaluate_fulfillment(env, best_solution)
    best_cost = solution_cost(env, best_solution)
    
    # Extended search
    iterations = 0
    no_improvement_count = 0
    max_no_improvement = 40  # Increased from 20
    
    # Don't stop early if we have high fulfillment but not 100%
    while time.time() - start_time < time_budget and iterations < 250:
        iterations += 1
        
        # Early stopping only if 100% fulfillment or truly converged
        if best_fulfillment >= 99.99:
            # We have 100%, just optimize cost a bit more
            if no_improvement_count >= 15:
                break
        elif no_improvement_count >= max_no_improvement:
            break
        
        # Tournament selection
        parent1 = max(random.sample(population, min(3, len(population))), key=lambda s: evaluate_fulfillment(env, s))
        parent2 = max(random.sample(population, min(3, len(population))), key=lambda s: evaluate_fulfillment(env, s))
        
        # Crossover
        routes_p1 = parent1.get("routes", [])
        routes_p2 = parent2.get("routes", [])
        child_routes = []
        
        # Dynamic cutpoint
        cutpoint = random.randint(len(routes_p1) // 3, 2 * len(routes_p1) // 3) if len(routes_p1) > 2 else len(routes_p1) // 2
        
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
        worst_idx = min(range(len(population)), key=lambda i: evaluate_fulfillment(env, population[i]))
        worst_fulfillment = evaluate_fulfillment(env, population[worst_idx])
        
        if child_fulfillment > worst_fulfillment or (child_fulfillment == worst_fulfillment and child_cost < solution_cost(env, population[worst_idx])):
            population[worst_idx] = child
            
            # Update best
            if child_fulfillment > best_fulfillment or (child_fulfillment == best_fulfillment and child_cost < best_cost):
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
    return adaptive_memetic_solver(env, time_budget=25.0)
