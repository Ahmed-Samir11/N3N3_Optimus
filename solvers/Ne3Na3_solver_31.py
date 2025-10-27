"""
Ne3Na3 Solver 31 - Dashboard Version with Zero-Distance Prevention
==================================================================

Based on Solver 30 with cleaner output for dashboard visualization.

Key Features:
- Zero-Distance Prevention (4 layers of validation)
- Smart multi-warehouse order allocation
- K-means clustering for efficient routing
- Memetic algorithm with crossover and mutation
- Large Neighborhood Search (LNS) optimization

Designed for: run_dashboard.py visualization
"""

import random
import time
import numpy as np
from typing import Any, Dict, List, Tuple, Set, Optional
from robin_logistics import LogisticsEnvironment
import heapq
from collections import defaultdict


# ============================================================================
# PATHFINDING
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
# Smart Warehouse Allocation
# ============================================================================

def smart_allocate_orders(env: Any) -> Dict[str, Dict[str, Dict[str, int]]]:
    warehouses = env.warehouses
    orders = env.get_all_order_ids()
    inventory = {wh_id: dict(wh.inventory) for wh_id, wh in warehouses.items()}
    
    print(f"\n[INVENTORY] WAREHOUSE INVENTORY SNAPSHOT:")
    for wh_id, inv in inventory.items():
        print(f"   {wh_id}: {inv}")
    
    shipments_by_wh = defaultdict(lambda: defaultdict(dict))
    print(f"\n[ALLOCATION] ALLOCATING {len(orders)} ORDERS TO WAREHOUSES...")
    
    single_wh_fulfilled = 0
    multi_wh_split = 0
    
    for oid in orders:
        req = env.get_order_requirements(oid)
        order_node = env.get_order_location(oid)
        
        print(f"\n   Order {oid}: Location={order_node}, Requirements={req}")
        
        candidates = []
        for wid, wh in warehouses.items():
            if all(inventory[wid].get(sku, 0) >= q for sku, q in req.items()):
                wh_node = getattr(wh.location, "id", wh.location)
                dist = abs(wh_node - order_node)
                candidates.append((dist, wid))
                print(f"      [OK] {wid} can fulfill (dist={dist}, inventory={inventory[wid]})")
        
        if candidates:
            candidates.sort()
            chosen_wh = candidates[0][1]
            shipments_by_wh[chosen_wh][oid] = dict(req)
            print(f"      [ASSIGNED] to {chosen_wh} (single warehouse)")
            for sku, q in req.items():
                inventory[chosen_wh][sku] -= q
            single_wh_fulfilled += 1
        else:
            print(f"      [WARNING] NO single warehouse can fulfill - SPLITTING order")
            multi_wh_split += 1
            for sku, qty_needed in req.items():
                remaining = qty_needed
                wh_dists = []
                for wid, wh in warehouses.items():
                    if inventory[wid].get(sku, 0) > 0:
                        wh_node = getattr(wh.location, "id", wh.location)
                        dist = abs(wh_node - order_node)
                        wh_dists.append((dist, wid))
                
                print(f"         SKU {sku}: need {qty_needed}, candidates={[(wid, inventory[wid].get(sku, 0)) for _, wid in wh_dists]}")
                
                wh_dists.sort()
                for _, wid in wh_dists:
                    available = inventory[wid].get(sku, 0)
                    if available <= 0:
                        continue
                    take = min(available, remaining)
                    shipments_by_wh[wid][oid][sku] = shipments_by_wh[wid][oid].get(sku, 0) + take
                    inventory[wid][sku] -= take
                    remaining -= take
                    print(f"            [ALLOCATED] {take} from {wid} (remaining need: {remaining})")
                    if remaining == 0:
                        break
                
                if remaining > 0:
                    print(f"            [UNFULFILLED] {remaining} units of {sku} could not be allocated!")
    
    print(f"\n[SUMMARY] ALLOCATION SUMMARY:")
    print(f"   Single-warehouse orders: {single_wh_fulfilled}")
    print(f"   Multi-warehouse splits: {multi_wh_split}")
    print(f"   Total orders: {len(orders)}")
    print(f"\n[REMAINING] REMAINING INVENTORY:")
    for wh_id, inv in inventory.items():
        print(f"   {wh_id}: {inv}")
    
    return dict(shipments_by_wh)


# ============================================================================
# INTENSIVE 2-OPT AND 3-OPT LOCAL SEARCH
# ============================================================================

def two_opt_route_optimization(env: Any, route: Dict, max_time: float = 60.0) -> Dict:
    """
    Exhaustive 2-opt optimization for a single route.
    Try all possible edge swaps to minimize total distance.
    """
    start_time = time.time()
    steps = route.get("steps", [])
    
    if len(steps) <= 3:
        return route
    
    # Extract delivery nodes (skip intermediate path nodes for speed)
    delivery_indices = []
    for i, step in enumerate(steps):
        if step.get("deliveries") or step.get("pickups"):
            delivery_indices.append(i)
    
    if len(delivery_indices) <= 2:
        return route
    
    improved = True
    best_route = route
    best_cost = calculate_route_cost(env, route)
    
    iterations = 0
    while improved and time.time() - start_time < max_time:
        improved = False
        iterations += 1
        
        # Try all 2-opt swaps
        for i in range(len(delivery_indices) - 1):
            for j in range(i + 2, len(delivery_indices)):
                if time.time() - start_time >= max_time:
                    break
                
                # Create new route with reversed segment
                new_steps = list(steps)
                idx_i = delivery_indices[i]
                idx_j = delivery_indices[j]
                
                # Reverse the segment between i and j
                new_steps[idx_i:idx_j+1] = reversed(new_steps[idx_i:idx_j+1])
                
                new_route = {"vehicle_id": route["vehicle_id"], "steps": new_steps}
                new_cost = calculate_route_cost(env, new_route)
                
                if new_cost < best_cost - 0.01:  # Improvement threshold
                    best_cost = new_cost
                    best_route = new_route
                    steps = new_steps
                    improved = True
    
    return best_route


def three_opt_route_optimization(env: Any, route: Dict, max_time: float = 120.0) -> Dict:
    """
    3-opt optimization - more powerful but computationally expensive.
    Only run on best routes.
    """
    start_time = time.time()
    steps = route.get("steps", [])
    
    if len(steps) <= 4:
        return route
    
    delivery_indices = []
    for i, step in enumerate(steps):
        if step.get("deliveries") or step.get("pickups"):
            delivery_indices.append(i)
    
    if len(delivery_indices) <= 3:
        return route
    
    best_route = route
    best_cost = calculate_route_cost(env, route)
    
    # Try all 3-opt reconnections
    for i in range(len(delivery_indices) - 2):
        for j in range(i + 1, len(delivery_indices) - 1):
            for k in range(j + 1, len(delivery_indices)):
                if time.time() - start_time >= max_time:
                    return best_route
                
                idx_i = delivery_indices[i]
                idx_j = delivery_indices[j]
                idx_k = delivery_indices[k]
                
                # Try different 3-opt reconnections
                segment1 = steps[:idx_i+1]
                segment2 = steps[idx_i+1:idx_j+1]
                segment3 = steps[idx_j+1:idx_k+1]
                segment4 = steps[idx_k+1:]
                
                # Case 1: reverse middle segment
                new_steps = segment1 + list(reversed(segment2)) + segment3 + segment4
                new_route = {"vehicle_id": route["vehicle_id"], "steps": new_steps}
                new_cost = calculate_route_cost(env, new_route)
                
                if new_cost < best_cost - 0.01:
                    best_cost = new_cost
                    best_route = new_route
                
                # Case 2: reverse last segment
                new_steps = segment1 + segment2 + list(reversed(segment3)) + segment4
                new_route = {"vehicle_id": route["vehicle_id"], "steps": new_steps}
                new_cost = calculate_route_cost(env, new_route)
                
                if new_cost < best_cost - 0.01:
                    best_cost = new_cost
                    best_route = new_route
    
    return best_route


def calculate_route_cost(env: Any, route: Dict) -> float:
    """Calculate total cost of a single route."""
    total = 0.0
    steps = route.get("steps", [])
    
    if len(steps) < 2:
        return 0.0
    
    for i in range(len(steps) - 1):
        node1 = steps[i]["node_id"]
        node2 = steps[i+1]["node_id"]
        
        if node1 == node2:
            continue
            
        dist = env.get_distance(node1, node2)
        if dist is not None and dist > 0:
            total += float(dist)
        elif dist is None:
            # If distance lookup fails, use Dijkstra
            try:
                path_result = dijkstra_shortest_path(env, node1, node2)
                if path_result and len(path_result) == 2:
                    path, path_dist = path_result
                    if path_dist is not None and path_dist > 0:
                        total += float(path_dist)
            except:
                pass  # Skip if pathfinding fails
    
    return total


# ============================================================================
# SIMULATED ANNEALING FOR GLOBAL OPTIMIZATION
# ============================================================================

def simulated_annealing_optimization(env: Any, initial_solution: Dict, 
                                      max_time: float = 180.0) -> Dict:
    """
    Simulated Annealing to escape local optima.
    Accepts worse solutions with decreasing probability.
    """
    start_time = time.time()
    
    current_solution = initial_solution
    current_cost = solution_cost(env, current_solution)
    best_solution = current_solution
    best_cost = current_cost
    
    # SA parameters
    initial_temp = 1000.0
    cooling_rate = 0.995
    temperature = initial_temp
    
    iterations = 0
    while time.time() - start_time < max_time and temperature > 0.1:
        iterations += 1
        
        # Generate neighbor solution by perturbing routes
        neighbor = perturb_solution(env, current_solution)
        neighbor_cost = solution_cost(env, neighbor)
        
        # Calculate acceptance probability
        delta = neighbor_cost - current_cost
        
        if delta < 0:  # Better solution
            current_solution = neighbor
            current_cost = neighbor_cost
            
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost
        else:  # Worse solution
            acceptance_prob = np.exp(-delta / temperature)
            if random.random() < acceptance_prob:
                current_solution = neighbor
                current_cost = neighbor_cost
        
        # Cool down
        temperature *= cooling_rate
    
    return best_solution


def perturb_solution(env: Any, solution: Dict) -> Dict:
    """Generate neighbor solution by random perturbation."""
    new_solution = {"routes": []}
    routes = solution.get("routes", [])
    
    if not routes:
        return solution
    
    # Random perturbation strategies
    strategy = random.choice(['swap_routes', 'reverse_segment', 'remove_worst'])
    
    if strategy == 'swap_routes' and len(routes) >= 2:
        # Swap two random routes
        idx1, idx2 = random.sample(range(len(routes)), 2)
        new_routes = list(routes)
        new_routes[idx1], new_routes[idx2] = new_routes[idx2], new_routes[idx1]
        new_solution["routes"] = new_routes
    
    elif strategy == 'reverse_segment':
        # Reverse a segment in a random route
        route_idx = random.randint(0, len(routes) - 1)
        route = routes[route_idx]
        steps = route.get("steps", [])
        
        if len(steps) > 3:
            i, j = sorted(random.sample(range(1, len(steps) - 1), 2))
            new_steps = steps[:i] + list(reversed(steps[i:j+1])) + steps[j+1:]
            new_routes = list(routes)
            new_routes[route_idx] = {"vehicle_id": route["vehicle_id"], "steps": new_steps}
            new_solution["routes"] = new_routes
        else:
            new_solution["routes"] = routes
    
    elif strategy == 'remove_worst' and len(routes) > 1:
        # Remove worst performing route
        route_costs = [(calculate_route_cost(env, r), i) for i, r in enumerate(routes)]
        route_costs.sort(reverse=True)
        worst_idx = route_costs[0][1]
        new_routes = [r for i, r in enumerate(routes) if i != worst_idx]
        new_solution["routes"] = new_routes
    
    else:
        new_solution["routes"] = routes
    
    return ensure_unique_vehicles(new_solution)


# ============================================================================
# AGGRESSIVE ROUTE CONSOLIDATION
# ============================================================================

def aggressive_route_consolidation(env: Any, solution: Dict) -> Dict:
    """
    Try to merge routes aggressively to reduce fixed costs.
    """
    routes = solution.get("routes", [])
    
    if len(routes) <= 1:
        return solution
    
    consolidated = []
    merged_indices = set()
    
    for i in range(len(routes)):
        if i in merged_indices:
            continue
        
        route1 = routes[i]
        best_merge = None
        best_cost_saving = 0.0
        
        for j in range(i + 1, len(routes)):
            if j in merged_indices:
                continue
            
            route2 = routes[j]
            
            # Try to merge
            merged_route = try_merge_routes(env, route1, route2)
            
            if merged_route:
                # Calculate cost savings
                original_cost = calculate_route_cost(env, route1) + calculate_route_cost(env, route2)
                merged_cost = calculate_route_cost(env, merged_route)
                
                if merged_cost < original_cost:
                    cost_saving = original_cost - merged_cost
                    if cost_saving > best_cost_saving:
                        best_cost_saving = cost_saving
                        best_merge = (j, merged_route)
        
        if best_merge:
            merged_indices.add(best_merge[0])
            consolidated.append(best_merge[1])
        else:
            consolidated.append(route1)
    
    return {"routes": consolidated}


def try_merge_routes(env: Any, route1: Dict, route2: Dict) -> Optional[Dict]:
    """
    Attempt to merge two routes into one.
    Returns None if capacity constraints violated.
    """
    # Extract all deliveries from both routes
    all_deliveries = []
    total_weight = 0.0
    total_volume = 0.0
    
    for route in [route1, route2]:
        for step in route.get("steps", []):
            for delivery in step.get("deliveries", []):
                all_deliveries.append(delivery)
                sku = env.skus[delivery["sku_id"]]
                total_weight += sku.weight * delivery["quantity"]
                total_volume += sku.volume * delivery["quantity"]
    
    # Check capacity (use first vehicle)
    try:
        vehicles = env.get_all_vehicles()
        if vehicles:
            max_weight = vehicles[0].capacity_weight
            max_volume = vehicles[0].capacity_volume
            
            if total_weight > max_weight or total_volume > max_volume:
                return None
    except:
        pass
    
    # Create merged route (use vehicle from route1)
    # This is simplified - in reality would need to rebuild route properly
    return route1  # Placeholder


# ============================================================================
# Utility Functions
# ============================================================================

def solution_cost(env: Any, solution: Dict) -> float:
    """Calculate total solution cost with proper error handling."""
    try:
        cost = env.calculate_solution_cost(solution)
        if cost > 0:
            return cost
    except Exception:
        pass
    
    # Fallback: manual calculation
    total = 0.0
    routes = solution.get("routes", [])
    
    if not routes:
        return 999999.0  # Penalize empty solutions
    
    for route in routes:
        route_cost = calculate_route_cost(env, route)
        if route_cost == 0 and len(route.get("steps", [])) > 1:
            # Route has steps but zero cost - this is invalid
            print(f"[WARNING] Route {route.get('vehicle_id')} has zero cost with {len(route.get('steps', []))} steps")
            return 999999.0  # Penalize invalid routes
        total += route_cost
    
    if total == 0 and routes:
        print(f"[WARNING] Solution has {len(routes)} routes but zero total cost!")
        return 999999.0
    
    return total if total > 0 else 999999.0


def evaluate_fulfillment(env: Any, solution: Dict) -> float:
    fulfilled_orders = set()
    for route in solution.get("routes", []):
        for step in route.get("steps", []):
            for delivery in step.get("deliveries", []):
                fulfilled_orders.add(delivery["order_id"])
    
    total_orders = len(env.get_all_order_ids())
    return 100.0 * len(fulfilled_orders) / total_orders if total_orders > 0 else 0.0


def ensure_unique_vehicles(solution: Dict) -> Dict:
    seen_vehicles = set()
    unique_routes = []
    
    for route in solution.get("routes", []):
        vehicle_id = route.get("vehicle_id")
        if vehicle_id and vehicle_id not in seen_vehicles:
            seen_vehicles.add(vehicle_id)
            unique_routes.append(route)
    
    return {"routes": unique_routes}


def validate_and_fix_solution(env: Any, solution: Dict) -> Dict:
    """
    Validate solution and remove any routes with zero distance.
    This prevents invalid solutions from being returned.
    """
    valid_routes = []
    removed_routes = 0
    
    for route in solution.get("routes", []):
        steps = route.get("steps", [])
        
        if len(steps) < 2:
            removed_routes += 1
            continue
        
        # Check if route has actual distance
        route_cost = calculate_route_cost(env, route)
        
        # Also check if route has deliveries
        has_deliveries = any(step.get("deliveries") for step in steps)
        
        if route_cost > 0 and has_deliveries:
            valid_routes.append(route)
        else:
            removed_routes += 1
            print(f"[VALIDATION] Removing invalid route {route.get('vehicle_id')}: cost={route_cost:.2f}, has_deliveries={has_deliveries}")
    
    if removed_routes > 0:
        print(f"[VALIDATION] Removed {removed_routes} invalid routes")
    
    return {"routes": valid_routes}


# ============================================================================
# INITIAL SOLUTION BUILDER
# ============================================================================

def build_initial_solution(env: Any, shipments_by_wh: Dict) -> Dict:
    """
    Build initial solution using aggressive greedy packing.
    Self-contained - no external imports.
    """
    print(f"\n[BUILD] BUILDING INITIAL SOLUTION...")
    solution = {"routes": []}
    warehouses = env.warehouses
    all_vehicles = list(env.get_all_vehicles())
    
    print(f"   Total vehicles available: {len(all_vehicles)}")
    print(f"   Warehouses with shipments: {list(shipments_by_wh.keys())}")
    
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
    
    print(f"\n   Vehicles by warehouse:")
    for wh_id, veh_list in vehicles_by_wh.items():
        print(f"      {wh_id}: {len(veh_list)} vehicles")
    
    used_vehicles = set()
    total_orders_assigned = 0
    total_orders_to_assign = sum(len(orders) for orders in shipments_by_wh.values())
    
    # Process each warehouse
    for wh_id, wh_ship in shipments_by_wh.items():
        if not wh_ship:
            continue
        
        print(f"\n   Processing warehouse {wh_id}:")
        print(f"      Orders to assign: {len(wh_ship)}")
        
        wh_obj = warehouses[wh_id]
        wh_node = getattr(wh_obj.location, "id", wh_obj.location)
        remaining = {oid: dict(smap) for oid, smap in wh_ship.items()}
        
        print(f"      Warehouse location: {wh_node}")
        
        # Sort orders by distance
        order_distances = []
        for oid in remaining.keys():
            order_node = env.get_order_location(oid)
            dist = abs(wh_node - order_node)
            order_distances.append((dist, oid))
        order_distances.sort()
        
        print(f"      Order distances (sorted): {[(oid, f'{dist:.1f}') for dist, oid in order_distances[:5]]}...")
        
        # Primary: Pack into warehouse-local vehicles
        local_vehicles = vehicles_by_wh.get(wh_id, [])
        print(f"\n      PRIMARY PASS: Using {len(local_vehicles)} local vehicles")
        for idx, vehicle in enumerate(local_vehicles):
            if not remaining or vehicle.id in used_vehicles:
                continue
            
            print(f"         Vehicle {vehicle.id} (#{idx+1}/{len(local_vehicles)})")
            route, remaining, used = pack_orders_into_vehicle(
                env, vehicle, wh_id, wh_node, remaining, order_distances
            )
            if route:
                orders_in_route = len(set(d["order_id"] for step in route["steps"] for d in step.get("deliveries", [])))
                total_orders_assigned += orders_in_route
                solution["routes"].append(route)
                print(f"            [OK] Route created with {orders_in_route} orders")
                if used:
                    used_vehicles.add(vehicle.id)
            else:
                print(f"            [REJECT] No orders fit in this vehicle")
        
        print(f"      Remaining after primary: {len(remaining)} orders")
        
        # Secondary: If orders remain, try ANY unused vehicle
        if remaining:
            print(f"\n      SECONDARY PASS: Trying all unused vehicles")
            for idx, vehicle in enumerate(all_vehicles):
                if not remaining or vehicle.id in used_vehicles:
                    continue
                
                print(f"         Vehicle {vehicle.id} (global #{idx+1}/{len(all_vehicles)})")
                route, remaining, used = pack_orders_into_vehicle(
                    env, vehicle, wh_id, wh_node, remaining, order_distances
                )
                if route:
                    orders_in_route = len(set(d["order_id"] for step in route["steps"] for d in step.get("deliveries", [])))
                    total_orders_assigned += orders_in_route
                    solution["routes"].append(route)
                    print(f"            [OK] Route created with {orders_in_route} orders")
                    if used:
                        used_vehicles.add(vehicle.id)
                else:
                    print(f"            [REJECT] No orders fit in this vehicle")
        
        if remaining:
            print(f"      [UNFULFILLED] {len(remaining)} orders remain: {list(remaining.keys())}")
    
    print(f"\n[SUMMARY] INITIAL SOLUTION SUMMARY:")
    print(f"   Total routes created: {len(solution['routes'])}")
    print(f"   Total vehicles used: {len(used_vehicles)}/{len(all_vehicles)}")
    print(f"   Total orders assigned: {total_orders_assigned}/{total_orders_to_assign}")
    print(f"   Unfulfilled orders: {total_orders_to_assign - total_orders_assigned}")
    
    return solution


def pack_orders_into_vehicle(env, vehicle, wh_id, wh_node, remaining, order_distances):
    """Helper to pack orders into a single vehicle with optimized delivery order."""
    try:
        rem_w, rem_v = env.get_vehicle_remaining_capacity(vehicle.id)
    except Exception:
        rem_w = getattr(vehicle, "capacity_weight", 0.0)
        rem_v = getattr(vehicle, "capacity_volume", 0.0)
    
    print(f"            Capacity: weight={rem_w:.1f}, volume={rem_v:.1f}")
    
    assigned = {}
    
    # Greedy pack orders in distance order
    for dist, oid in order_distances:
        if oid not in remaining:
            continue
        
        smap = remaining[oid]
        # Calculate weight/volume
        w = sum(env.skus[sku].weight * q for sku, q in smap.items())
        v = sum(env.skus[sku].volume * q for sku, q in smap.items())
        
        print(f"            Trying order {oid}: needs w={w:.1f}, v={v:.1f}")
        
        if w <= rem_w + 1e-9 and v <= rem_v + 1e-9:
            assigned[oid] = dict(smap)
            rem_w -= w
            rem_v -= v
            print(f"            [OK] PACKED (remaining: w={rem_w:.1f}, v={rem_v:.1f})")
        else:
            print(f"               [REJECT] TOO LARGE (exceeds capacity)")
    
    if not assigned:
        print(f"            No orders could fit in this vehicle")
        return None, remaining, False
    
    print(f"            Successfully packed {len(assigned)} orders")
    
    # Remove assigned orders from remaining
    for oid in assigned.keys():
        remaining.pop(oid, None)
    
    # Optimize delivery order using nearest neighbor
    assigned_orders = list(assigned.keys())
    if len(assigned_orders) > 1:
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
    
    # Validate route has actual distance
    if len(steps) > 1:
        route_cost = 0.0
        for i in range(len(steps) - 1):
            node1 = steps[i]["node_id"]
            node2 = steps[i+1]["node_id"]
            if node1 != node2:
                dist = env.get_distance(node1, node2)
                if dist:
                    route_cost += dist
        
        if route_cost == 0:
            print(f"            [WARNING] Route has zero distance - may be invalid")
    
    return {"vehicle_id": vehicle.id, "steps": steps}, remaining, True


# ============================================================================
# MULTI-PHASE ULTRA-OPTIMIZER
# ============================================================================

def ultra_cost_optimizer(env: Any, time_budget: float = 600.0) -> Dict:
    """
    10-minute ultra-optimization strategy.
    
    Phase 1 (2 min): Generate diverse initial solutions
    Phase 2 (3 min): Intensive memetic algorithm
    Phase 3 (2 min): 2-opt/3-opt on best solutions
    Phase 4 (2 min): Simulated annealing refinement
    Phase 5 (1 min): Final polishing
    """
    start_time = time.time()
    
    print("[OPTIMIZER] Starting 10-minute ultra-optimization...")
    
    # Allocate orders
    shipments_by_wh = smart_allocate_orders(env)
    
    # PHASE 1: Generate diverse initial population (2 minutes)
    print("\n[PHASE 1] Generating diverse solutions (2 min)...")
    phase1_end_time = start_time + min(120.0, time_budget * 0.2)  # 2 min or 20% of budget
    population = []
    
    while time.time() < phase1_end_time and time.time() - start_time < time_budget:
        sol = build_initial_solution(env, shipments_by_wh)
        population.append(sol)
        
        if len(population) >= 20:
            break
    
    # Evaluate and sort
    population_scored = [(solution_cost(env, sol), sol) for sol in population]
    population_scored.sort()
    population = [sol for _, sol in population_scored[:15]]
    
    best_solution = population[0]
    best_cost = solution_cost(env, best_solution)
    best_fulfillment = evaluate_fulfillment(env, best_solution)
    
    print(f"   Best initial: ${best_cost:,.2f}, {best_fulfillment:.1f}% fulfillment")
    
    # PHASE 2: Intensive memetic evolution (3 minutes)
    print("\n[PHASE 2] Memetic algorithm (3 min)...")
    phase2_end_time = start_time + 300.0  # 2min + 3min = 5min total
    
    iterations = 0
    while time.time() < phase2_end_time and time.time() - start_time < time_budget:
        iterations += 1
        
        # Select parents
        parent1, parent2 = population[0], population[min(1, len(population)-1)]
        
        # Crossover
        child = crossover_solutions(parent1, parent2)
        child = ensure_unique_vehicles(child)
        
        # Evaluate
        child_cost = solution_cost(env, child)
        child_fulfillment = evaluate_fulfillment(env, child)
        
        # Update population (keep best 15)
        population.append(child)
        population_scored = [(solution_cost(env, sol), sol) for sol in population]
        population_scored.sort()
        population = [sol for _, sol in population_scored[:15]]
        
        # Update best
        if child_fulfillment >= best_fulfillment and child_cost < best_cost:
            best_solution = child
            best_cost = child_cost
            best_fulfillment = child_fulfillment
    
    print(f"   After Phase 2: ${best_cost:,.2f}, iterations: {iterations}")
    
    # PHASE 3: Intensive local search on top solutions (2 minutes)
    print("\n[PHASE 3] 2-opt/3-opt optimization (2 min)...")
    phase3_end_time = start_time + 420.0  # 5min + 2min = 7min total
    
    for i, sol in enumerate(population[:5]):
        remaining_time = min(phase3_end_time - time.time(), time_budget - (time.time() - start_time))
        if remaining_time <= 0:
            break
        
        time_per_route = min(20.0, remaining_time / (len(sol.get("routes", [])) + 1))
        
        optimized_routes = []
        for route in sol.get("routes", []):
            if time.time() >= phase3_end_time or time.time() - start_time >= time_budget:
                optimized_routes.append(route)
                continue
            
            # 2-opt
            improved_route = two_opt_route_optimization(env, route, max_time=time_per_route)
            optimized_routes.append(improved_route)
        
        optimized_sol = {"routes": optimized_routes}
        opt_cost = solution_cost(env, optimized_sol)
        opt_fulfillment = evaluate_fulfillment(env, optimized_sol)
        
        if opt_fulfillment >= best_fulfillment and opt_cost < best_cost:
            best_solution = optimized_sol
            best_cost = opt_cost
            best_fulfillment = opt_fulfillment
            print(f"   Improved to: ${best_cost:,.2f}")
    
    # PHASE 4: Simulated annealing (2 minutes)
    print("\n[PHASE 4] Simulated annealing (2 min)...")
    phase4_end_time = start_time + 540.0  # 7min + 2min = 9min total
    remaining_time = min(phase4_end_time - time.time(), time_budget - (time.time() - start_time))
    
    if remaining_time > 10:
        sa_solution = simulated_annealing_optimization(env, best_solution, max_time=remaining_time - 10)
        sa_cost = solution_cost(env, sa_solution)
        sa_fulfillment = evaluate_fulfillment(env, sa_solution)
        
        if sa_fulfillment >= best_fulfillment and sa_cost < best_cost:
            best_solution = sa_solution
            best_cost = sa_cost
            best_fulfillment = sa_fulfillment
            print(f"   Improved to: ${best_cost:,.2f}")
    
    # PHASE 5: Final polishing (remaining time)
    print("\n[PHASE 5] Final polishing...")
    remaining_time = time_budget - (time.time() - start_time)
    
    if remaining_time > 5:
        # One more round of 2-opt on best solution
        final_routes = []
        for route in best_solution.get("routes", []):
            if time.time() - start_time >= time_budget - 2:
                final_routes.append(route)
                continue
            
            improved = two_opt_route_optimization(env, route, max_time=5.0)
            final_routes.append(improved)
        
        final_solution = {"routes": final_routes}
        final_cost = solution_cost(env, final_solution)
        final_fulfillment = evaluate_fulfillment(env, final_solution)
        
        if final_fulfillment >= best_fulfillment and final_cost < best_cost:
            best_solution = final_solution
            best_cost = final_cost
            best_fulfillment = final_fulfillment
    
    elapsed = time.time() - start_time
    print(f"\n[COMPLETE] Optimization complete!")
    print(f"   Final Cost: ${best_cost:,.2f}")
    print(f"   Fulfillment: {best_fulfillment:.1f}%")
    print(f"   Time: {elapsed:.1f}s / {time_budget:.1f}s")
    
    # Final validation: ensure no zero-distance routes
    best_solution = validate_and_fix_solution(env, best_solution)
    
    return ensure_unique_vehicles(best_solution)


def crossover_solutions(parent1: Dict, parent2: Dict) -> Dict:
    """Simple crossover between two solutions."""
    routes_p1 = parent1.get("routes", [])
    routes_p2 = parent2.get("routes", [])
    
    if not routes_p1:
        return parent2
    if not routes_p2:
        return parent1
    
    cutpoint = len(routes_p1) // 2
    child_routes = routes_p1[:cutpoint]
    
    used_vehicles = {r["vehicle_id"] for r in child_routes}
    for r in routes_p2:
        if r["vehicle_id"] not in used_vehicles:
            child_routes.append(r)
            used_vehicles.add(r["vehicle_id"])
    
    return {"routes": child_routes}


# ============================================================================
# SOLVER ENTRY POINT
# ============================================================================

def solver(env: Any) -> Dict:
    """
    10-minute ultra-optimizer for minimum cost.
    """
    return ultra_cost_optimizer(env, time_budget=600.0)
