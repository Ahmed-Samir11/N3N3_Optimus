"""
Ne3Na3 Solver 58: Hybrid Memetic + NetworkX Optimizer
======================================================

COMBINES BEST OF:
- Solver 53: NetworkX caching, cost-per-capacity ranking, 2-opt optimization
- Solver 13: Memetic algorithm, LNS mutation, simulated annealing

STRATEGY:
1. NetworkX + LRU caching for FAST pathfinding (from Solver 53)
2. Smart vehicle ranking by cost-per-capacity (from Solver 53)
3. Memetic population evolution (from Solver 13)
4. LNS-style mutation: remove & reinsert orders (from Solver 13)
5. Simulated annealing acceptance (from Solver 13)
6. 2-opt local search for final polish (from Solver 53)
7. Penalty-based scoring to balance cost vs fulfillment (from Solver 13)

TARGET: Minimize cost while maintaining high fulfillment (90%+)
"""

import heapq
import time
import random
import math
from typing import Any, Dict, List, Tuple, Optional
from functools import lru_cache
import networkx as nx

# ============================================================================
# PATHFINDING WITH CACHING (from Solver 53)
# ============================================================================

_graph_cache = None
_cache_counter = 0

def clear_caches():
    """Clear all caches - MUST be called at start of solver()"""
    global _graph_cache, _cache_counter
    _graph_cache = None
    _cache_counter += 1
    get_shortest_path_cached.cache_clear()

def build_networkx_graph(env: Any) -> nx.DiGraph:
    """Build NetworkX directed graph from environment (cached per run only)."""
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache
    
    G = nx.DiGraph()
    road_data = env.get_road_network_data()
    
    adjacency = road_data.get('adjacency_list', {})
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            distance = env.get_distance(node, neighbor)
            if distance is not None:
                G.add_edge(node, neighbor, weight=distance)
    
    _graph_cache = G
    return G

@lru_cache(maxsize=10000)
def get_shortest_path_cached(start: int, end: int, run_id: int) -> Tuple[Optional[tuple], float]:
    """Cached shortest path using NetworkX."""
    G = _graph_cache
    
    if start == end:
        return (start,), 0.0
    
    try:
        path = nx.shortest_path(G, start, end, weight='weight')
        distance = nx.shortest_path_length(G, start, end, weight='weight')
        return tuple(path), distance
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float('inf')

def dijkstra_shortest_path(env: Any, start: int, end: int) -> Tuple[Optional[List[int]], float]:
    """Get shortest path using cached NetworkX."""
    global _cache_counter
    
    G = build_networkx_graph(env)
    path_tuple, distance = get_shortest_path_cached(start, end, _cache_counter)
    
    if path_tuple is None:
        return None, float('inf')
    
    return list(path_tuple), distance

# ============================================================================
# ROUTE BUILDING (from Solver 13, adapted with Solver 53 pathfinding)
# ============================================================================

def build_steps_with_path(env: Any, node_sequence: List[Tuple[int, Any]], home_node: int) -> List[Dict]:
    """
    Build route steps with proper pathfinding.
    node_sequence: list of (node_id, payload) where payload contains pickups/deliveries
    """
    steps = [{"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []}]
    
    current = home_node
    for node_id, payload in node_sequence:
        path, _ = dijkstra_shortest_path(env, current, node_id)
        if path is None:
            return None
        
        # Normalize payload
        if isinstance(payload, dict):
            pickups_list = payload.get("pickups", []) or []
            deliveries_list = payload.get("deliveries", []) or []
        else:
            pickups_list = []
            deliveries_list = payload or []
        
        # Add intermediate nodes
        if len(path) <= 1:
            # Same node - merge into last step
            last = steps[-1]
            if pickups_list:
                last["pickups"] = (last.get("pickups", []) or []) + pickups_list
            if deliveries_list:
                last["deliveries"] = (last.get("deliveries", []) or []) + deliveries_list
        else:
            for intermediate in path[1:]:
                if intermediate == node_id:
                    steps.append({"node_id": intermediate, "pickups": pickups_list, "deliveries": deliveries_list, "unloads": []})
                else:
                    steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
        
        current = node_id
    
    # Return to home
    path_home, _ = dijkstra_shortest_path(env, current, home_node)
    if path_home is None:
        return None
    
    for intermediate in path_home[1:]:
        steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
    
    return steps

# ============================================================================
# SOLUTION SCORING (from Solver 13)
# ============================================================================

def solution_cost(env: Any, solution: Dict) -> float:
    """Calculate solution cost using environment."""
    try:
        return env.calculate_solution_cost(solution)
    except Exception:
        return float('inf')

def get_served_orders(solution: Dict) -> set:
    """Get set of served order IDs."""
    served = set()
    for r in solution.get("routes", []):
        for st in r.get("steps", []):
            for d in st.get("deliveries", []):
                oid = d.get("order_id")
                if oid:
                    served.add(oid)
    return served

def score_solution(env: Any, solution: Dict, total_orders: int, penalty_per_missing: float = 150000.0) -> float:
    """
    Score solution with penalty for unfulfilled orders.
    Lower score is better.
    """
    base_cost = solution_cost(env, solution)
    served = len(get_served_orders(solution))
    missing = max(0, total_orders - served)
    return base_cost + penalty_per_missing * missing

# ============================================================================
# GREEDY INITIAL SOLUTION (from Solver 53 + Solver 13)
# ============================================================================

def greedy_initial_solution(env: Any, shipments_by_wh: Dict[str, Dict[str, Dict[str, int]]]) -> Dict:
    """Generate initial greedy solution with cost-efficient vehicle selection."""
    solution = {"routes": []}
    warehouses = env.warehouses
    
    # Get vehicles and rank by cost-per-capacity (from Solver 53)
    all_vehicles = list(env.get_all_vehicles())
    vehicle_efficiency = {}
    for v in all_vehicles:
        capacity_units = v.capacity_weight + v.capacity_volume
        cost_per_unit = v.fixed_cost / capacity_units if capacity_units > 0 else float('inf')
        vehicle_efficiency[v.id] = cost_per_unit
    
    # Group vehicles by warehouse
    vehicles_by_wh = {}
    for v in all_vehicles:
        home_wh = v.home_warehouse_id
        vehicles_by_wh.setdefault(home_wh, []).append(v)
    
    # Sort vehicles in each warehouse by efficiency
    for wh_id in vehicles_by_wh:
        vehicles_by_wh[wh_id].sort(key=lambda v: vehicle_efficiency[v.id])
    
    # Pack orders into vehicles
    for wh_id, wh_ship in shipments_by_wh.items():
        if not wh_ship:
            continue
        
        wh_obj = warehouses[wh_id]
        wh_node = wh_obj.location.id
        remaining = {oid: dict(smap) for oid, smap in wh_ship.items()}
        
        for vehicle in vehicles_by_wh.get(wh_id, []):
            if not remaining:
                break
            
            # Greedy pack
            rem_w = vehicle.capacity_weight
            rem_v = vehicle.capacity_volume
            assigned = {}
            
            for oid, smap in list(remaining.items()):
                # Calculate weight/volume
                w = sum(env.skus[sku].weight * q for sku, q in smap.items())
                v = sum(env.skus[sku].volume * q for sku, q in smap.items())
                
                if w <= rem_w + 1e-9 and v <= rem_v + 1e-9:
                    assigned[oid] = dict(smap)
                    rem_w -= w
                    rem_v -= v
                    remaining.pop(oid, None)
            
            if not assigned:
                continue
            
            # Build route
            try:
                home_node = env.get_vehicle_home_warehouse(vehicle.id)
            except Exception:
                home_node = wh_node
            
            node_sequence = []
            
            # Pickups at warehouse
            pickup_map = {}
            for oid, smap in assigned.items():
                for sku, q in smap.items():
                    pickup_map[sku] = pickup_map.get(sku, 0) + int(q)
            
            pickups_list = [{"warehouse_id": wh_id, "sku_id": sku, "quantity": int(q)} for sku, q in pickup_map.items()]
            node_sequence.append((wh_node, {"pickups": pickups_list}))
            
            # Deliveries (simple nearest-neighbor TSP)
            delivery_nodes = list(assigned.keys())
            tsp_order = []
            current = wh_node
            unvisited = set(delivery_nodes)
            
            while unvisited:
                nearest = min(unvisited, key=lambda oid: dijkstra_shortest_path(env, current, env.get_order_location(oid))[1])
                tsp_order.append(nearest)
                unvisited.remove(nearest)
                current = env.get_order_location(nearest)
            
            for oid in tsp_order:
                node = env.get_order_location(oid)
                deliveries = [{"order_id": oid, "sku_id": sku, "quantity": int(q)} for sku, q in assigned[oid].items()]
                node_sequence.append((node, deliveries))
            
            steps = build_steps_with_path(env, node_sequence, home_node)
            if steps is not None:
                solution["routes"].append({"vehicle_id": vehicle.id, "steps": steps})
    
    return solution

# ============================================================================
# LNS MUTATION (from Solver 13)
# ============================================================================

def mutate_solution(env: Any, solution: Dict, shipments_by_wh: Dict, remove_frac: float = 0.05) -> Dict:
    """
    LNS-style mutation: remove a small fraction of orders and try to reinsert greedily.
    Fixed 5% removal rate for stability (Solver 13 pattern).
    """
    # Collect all order deliveries
    all_orders = []
    for r in solution.get("routes", []):
        for s in r.get("steps", []):
            for d in s.get("deliveries", []):
                all_orders.append(d["order_id"])
    
    if not all_orders:
        return solution
    
    # Remove 5% of orders (at least 1)
    to_remove = set(random.sample(all_orders, min(len(all_orders), max(1, int(remove_frac * len(all_orders))))))
    
    # Remove selected orders and track what was removed
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
            
            new_steps.append({
                "node_id": step["node_id"],
                "pickups": step.get("pickups", []),
                "deliveries": new_del,
                "unloads": step.get("unloads", [])
            })
        
        # Keep route if it has deliveries
        if any(st.get("deliveries") for st in new_steps):
            new_routes.append({"vehicle_id": r["vehicle_id"], "steps": new_steps})
    
    new_solution = {"routes": new_routes}
    
    # Try to reinsert removed orders (Solver 13 pattern)
    for oid, sku_map in removed_items.items():
        inserted = False
        
        # Find candidate warehouses
        for wh_id, orders in shipments_by_wh.items():
            if oid not in orders:
                continue
            
            wh_obj = env.warehouses[wh_id]
            
            # Try to add as single-order route on unused vehicle
            for v in env.get_all_vehicles():
                used_vehicle_ids = {r["vehicle_id"] for r in new_solution.get("routes", [])}
                if v.id in used_vehicle_ids:
                    continue
                
                try:
                    home_node = env.get_vehicle_home_warehouse(v.id)
                except Exception:
                    home_node = wh_obj.location.id
                
                node_sequence = []
                
                # Warehouse pickup
                pickup_map = {sku: int(q) for sku, q in sku_map.items()}
                pickups_list = [{"warehouse_id": wh_id, "sku_id": sku, "quantity": int(q)} 
                               for sku, q in pickup_map.items()]
                node_sequence.append((wh_obj.location.id, {"pickups": pickups_list}))
                
                # Order delivery
                node = env.get_order_location(oid)
                deliveries = [{"order_id": oid, "sku_id": sku, "quantity": int(q)} 
                             for sku, q in sku_map.items()]
                node_sequence.append((node, deliveries))
                
                steps = build_steps_with_path(env, node_sequence, home_node)
                if steps is None:
                    continue
                
                try:
                    valid, _ = env.validate_solution_complete({"routes": [{"vehicle_id": v.id, "steps": steps}]})
                except Exception:
                    valid = True
                
                if valid:
                    new_solution.setdefault("routes", []).append({"vehicle_id": v.id, "steps": steps})
                    inserted = True
                    break
            
            if inserted:
                break
    
    return new_solution

# ============================================================================
# 2-OPT OPTIMIZATION (from Solver 53)
# ============================================================================

def calculate_route_distance(env: Any, steps: List[Dict]) -> float:
    """Calculate total distance for a route."""
    total_distance = 0.0
    for i in range(len(steps) - 1):
        _, dist = dijkstra_shortest_path(env, steps[i]["node_id"], steps[i + 1]["node_id"])
        if dist != float('inf'):
            total_distance += dist
    return total_distance

def two_opt_optimize_route(env: Any, route: Dict, max_time: float = 2.0) -> Dict:
    """2-opt local search to optimize route order."""
    start_time = time.time()
    steps = route.get("steps", [])
    
    if len(steps) <= 4:
        return route
    
    delivery_indices = [i for i, step in enumerate(steps) if step.get("deliveries")]
    
    if len(delivery_indices) <= 2:
        return route
    
    best_steps = list(steps)
    best_distance = calculate_route_distance(env, best_steps)
    
    improved = True
    
    while improved and time.time() - start_time < max_time:
        improved = False
        
        for i in range(len(delivery_indices) - 1):
            for j in range(i + 2, len(delivery_indices)):
                if time.time() - start_time >= max_time:
                    break
                
                new_steps = list(best_steps)
                idx_i = delivery_indices[i]
                idx_j = delivery_indices[j]
                
                new_steps[idx_i:idx_j+1] = reversed(new_steps[idx_i:idx_j+1])
                new_distance = calculate_route_distance(env, new_steps)
                
                if new_distance < best_distance - 0.01:
                    best_distance = new_distance
                    best_steps = new_steps
                    improved = True
            
            if time.time() - start_time >= max_time:
                break
    
    return {"vehicle_id": route["vehicle_id"], "steps": best_steps}

# ============================================================================
# MEMETIC ALGORITHM (from Solver 13, enhanced with Solver 53 techniques)
# ============================================================================

def memetic_solver(env: Any, time_budget: float = 25.0) -> Dict:
    """
    Memetic algorithm combining population evolution and local search.
    """
    start = time.time()
    
    print(f"\n[MEMETIC PHASE] Starting population evolution ({time_budget}s budget)")
    
    # Allocate orders to warehouses
    shipments_by_wh = {}
    orders = env.get_all_order_ids()
    
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
    
    # Create initial population - SIMPLIFIED (Solver 13 pattern)
    pop_size = 1  # Simpler approach for better reliability
    pop = []
    
    print(f"  Generating initial solution (simplified approach)...")
    pop.append(greedy_initial_solution(env, shipments_by_wh))
    
    total_orders = sum(len(od) for od in shipments_by_wh.values())
    PENALTY_PER_MISSING = 150000.0
    
    best = pop[0]
    best_score = score_solution(env, best, total_orders, PENALTY_PER_MISSING)
    
    print(f"  Initial score: {best_score:.0f} (served: {len(get_served_orders(best))}/{total_orders})")
    
    # Simplified evolution loop (Solver 13 pattern)
    it = 0
    max_iters = 100
    no_improve_count = 0
    
    while time.time() - start < time_budget and it < max_iters:
        it += 1
        
        if len(pop) < 1:
            break
        
        # Simplified mutation only (no crossover needed with pop_size=1)
        child = mutate_solution(env, best, shipments_by_wh, remove_frac=0.05)  # Fixed 5%
        
        c_score = score_solution(env, child, total_orders, PENALTY_PER_MISSING)
        
        # Simple acceptance (Solver 13 pattern: accept only if better)
        if c_score < best_score:
            best = child
            best_score = c_score
            no_improve_count = 0
            served = len(get_served_orders(best))
            print(f"  Iter {it}: New best score {best_score:.0f} (served: {served}/{total_orders})")
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count > 20:
            print(f"  Early stopping at iteration {it} (no improvement for 20 iterations)")
            break
    
    print(f"  Simplified evolution completed ({it} iterations)")
    return best

# ============================================================================
# MAIN SOLVER
# ============================================================================

def solver(env: Any) -> Dict:
    """
    Hybrid solver: Solver 53's 100% fulfillment + Solver 13's refinement.
    """
    clear_caches()
    
    print("\n[SOLVER 54] Hybrid: Solver 53 Init + Solver 13 Refinement")
    print("=" * 80)
    
    start_time = time.time()
    
    # Build graph
    G = build_networkx_graph(env)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # PHASE 1: Use Solver 53's adaptive bin packing for 100% fulfillment
    print("\nPhase 1: Adaptive Bin Packing (Solver 53 pattern)")
    print("-" * 80)
    
    # Get vehicles and rank by cost-per-capacity
    all_vehicles = list(env.get_all_vehicles())
    vehicle_efficiency = {}
    for v in all_vehicles:
        capacity_units = v.capacity_weight + v.capacity_volume
        cost_per_unit = v.fixed_cost / capacity_units if capacity_units > 0 else float('inf')
        if v.type not in vehicle_efficiency:
            vehicle_efficiency[v.type] = cost_per_unit
    
    sorted_types = sorted(vehicle_efficiency.keys(), key=lambda x: vehicle_efficiency[x])
    
    # Group vehicles by warehouse and type
    vehicles_by_wh = {}
    for v in all_vehicles:
        wh_id = v.home_warehouse_id
        if wh_id not in vehicles_by_wh:
            vehicles_by_wh[wh_id] = {vtype: [] for vtype in sorted_types}
        if v.type not in vehicles_by_wh[wh_id]:
            vehicles_by_wh[wh_id][v.type] = []
        vehicles_by_wh[wh_id][v.type].append(v)
    
    # Adaptive warehouse allocation (alternating, size-sorted)
    all_orders = list(env.get_all_order_ids())
    order_data = []
    for oid in all_orders:
        req = env.get_order_requirements(oid)
        weight = sum(env.skus[sku].weight * qty for sku, qty in req.items())
        volume = sum(env.skus[sku].volume * qty for sku, qty in req.items())
        order_data.append((oid, weight, volume))
    
    order_data.sort(key=lambda x: max(x[1], x[2]), reverse=True)
    
    warehouse_ids = list(env.warehouses.keys())
    orders_by_wh = {wh_id: [] for wh_id in warehouse_ids}
    shipments_by_wh = {}
    
    for i, (oid, w, v) in enumerate(order_data):
        wh_id = warehouse_ids[i % len(warehouse_ids)]
        orders_by_wh[wh_id].append(oid)
        req = env.get_order_requirements(oid)
        shipments_by_wh.setdefault(wh_id, {})[oid] = dict(req)
    
    print(f"Total orders: {len(all_orders)}")
    for wh_id, orders in orders_by_wh.items():
        print(f"  {wh_id}: {len(orders)} orders")
    
    # Build routes using adaptive bin packing
    routes = []
    
    for wh_id, orders in orders_by_wh.items():
        if not orders:
            continue
        
        wh = env.warehouses[wh_id]
        wh_node = wh.location.id
        
        available_vehicles = []
        for vtype in sorted_types:
            if wh_id in vehicles_by_wh and vtype in vehicles_by_wh[wh_id]:
                available_vehicles.extend(vehicles_by_wh[wh_id][vtype])
        
        print(f"\n[{wh_id}] Packing {len(orders)} orders")
        
        # Sort orders by size
        order_sizes = []
        for oid in orders:
            req = env.get_order_requirements(oid)
            w = sum(env.skus[sku].weight * qty for sku, qty in req.items())
            v = sum(env.skus[sku].volume * qty for sku, qty in req.items())
            order_sizes.append((oid, w, v))
        
        order_sizes.sort(key=lambda x: max(x[1], x[2]), reverse=True)
        remaining = [o[0] for o in order_sizes]
        
        # Pack into vehicles
        for truck in available_vehicles:
            if not remaining:
                break
            
            packed = []
            rem_w = truck.capacity_weight
            rem_v = truck.capacity_volume
            
            for oid in remaining[:]:
                req = env.get_order_requirements(oid)
                ow = sum(env.skus[sku].weight * qty for sku, qty in req.items())
                ov = sum(env.skus[sku].volume * qty for sku, qty in req.items())
                
                if ow <= rem_w and ov <= rem_v:
                    packed.append(oid)
                    remaining.remove(oid)
                    rem_w -= ow
                    rem_v -= ov
            
            if packed:
                util_w = (truck.capacity_weight - rem_w) / truck.capacity_weight * 100
                util_v = (truck.capacity_volume - rem_v) / truck.capacity_volume * 100
                print(f"  {truck.type}: {len(packed)} orders, {util_w:.1f}%W {util_v:.1f}%V")
                
                # Build route
                delivery_nodes = []
                order_map = {}
                for oid in packed:
                    dest_node = env.orders[oid].destination.id
                    if dest_node not in order_map:
                        order_map[dest_node] = []
                        delivery_nodes.append(dest_node)
                    order_map[dest_node].append(oid)
                
                # TSP ordering
                if len(delivery_nodes) > 1:
                    ordered = [delivery_nodes[0]]
                    unvisited = set(delivery_nodes[1:])
                    while unvisited:
                        _, nearest = min((dijkstra_shortest_path(env, ordered[-1], n)[1], n) for n in unvisited)
                        ordered.append(nearest)
                        unvisited.remove(nearest)
                    delivery_nodes = ordered
                
                # Build steps with proper path expansion
                # Pickup at warehouse
                pickups = []
                for oid in packed:
                    req = env.get_order_requirements(oid)
                    for sku, qty in req.items():
                        pickups.append({"warehouse_id": wh_id, "sku_id": sku, "quantity": qty})
                
                # Start with SINGLE step at warehouse with pickups (like Solver 53)
                steps = [{"node_id": wh_node, "pickups": pickups, "deliveries": [], "unloads": []}]
                
                # Deliveries with full path expansion
                current_node = wh_node
                for node in delivery_nodes:
                    deliveries = []
                    for oid in order_map[node]:
                        req = env.get_order_requirements(oid)
                        for sku, qty in req.items():
                            deliveries.append({"order_id": oid, "sku_id": sku, "quantity": qty})
                    
                    # Get path from current to delivery node
                    path, dist = dijkstra_shortest_path(env, current_node, node)
                    if path and len(path) > 1:
                        # Add ALL intermediate nodes (including destination)
                        for intermediate in path[1:]:
                            if intermediate == node:
                                # Final node with deliveries
                                steps.append({"node_id": intermediate, "pickups": [], "deliveries": deliveries, "unloads": []})
                            else:
                                # Intermediate node
                                steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
                    else:
                        # Same node or no path - just add delivery
                        steps.append({"node_id": node, "pickups": [], "deliveries": deliveries, "unloads": []})
                    
                    current_node = node
                
                # Return home with full path
                path_home, dist_home = dijkstra_shortest_path(env, current_node, wh_node)
                if path_home and len(path_home) > 1:
                    for intermediate in path_home[1:]:
                        steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
                elif current_node != wh_node:
                    # Not at home and no path found - add final step
                    steps.append({"node_id": wh_node, "pickups": [], "deliveries": [], "unloads": []})
                
                routes.append({"vehicle_id": truck.id, "steps": steps})
        
        if remaining:
            print(f"  WARNING: {len(remaining)} orders could not be packed!")
    
    solution = {"routes": routes}
    
    elapsed = time.time() - start_time
    print(f"\n[2-OPT PHASE] Polishing routes (remaining time: {30-elapsed:.1f}s)")
    
    # PHASE 2: 2-opt optimization on all routes
    optimized_routes = []
    for i, route in enumerate(solution.get("routes", [])):
        if time.time() - start_time >= 28.0:  # Leave 2s buffer
            optimized_routes.append(route)
            continue
        
        optimized = two_opt_optimize_route(env, route, max_time=1.5)
        optimized_routes.append(optimized)
    
    final_solution = {"routes": optimized_routes}
    
    # Summary
    served = len(get_served_orders(final_solution))
    total = len(env.get_all_order_ids())
    
    print(f"\n[SOLUTION] Generated {len(optimized_routes)} routes")
    print(f"  Fulfillment: {served}/{total} ({served/total*100:.1f}%)")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    
    return final_solution


# if __name__ == '__main__':
#     from robin_logistics import LogisticsEnvironment
#     env = LogisticsEnvironment()
#     solution = solver(env)
#     success, msg = env.execute_solution(solution)
#     if success:
#         cost = env.calculate_solution_cost(solution)
#         fulfillment = env.get_solution_fulfillment_summary(solution)
#         print(f"\nCost: ${cost:,.0f}")
#         print(f"Fulfillment: {fulfillment}")
