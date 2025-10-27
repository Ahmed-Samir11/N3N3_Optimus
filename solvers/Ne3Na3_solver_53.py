"""
Ne3Na3 Solver 53: Adaptive High-Performance Solver
===================================================

STRATEGY:
1. NetworkX + LRU caching for FAST pathfinding (cleared per run)
2. Smart vehicle selection based on cost-per-capacity ratio
3. Adaptive packing: Fill large vehicles first, use small for overflow
4. 2-opt local search to optimize route distances
5. Generic algorithm - works across ALL scenarios

COMPLIANCE:
- NO persistent caching between solver() calls
- LRU cache cleared at start of each solver() run
- All distances computed fresh from environment per run
"""

import heapq
import time
from typing import Any, Dict, List, Tuple, Optional
from functools import lru_cache
import networkx as nx

# Global cache (MUST be reset at start of each solver run)
_graph_cache = None
_cache_counter = 0  # Used to invalidate @lru_cache between runs

def clear_caches():
    """Clear all caches - MUST be called at start of solver()"""
    global _graph_cache, _cache_counter
    _graph_cache = None
    _cache_counter += 1  # Invalidates all @lru_cache calls
    get_shortest_path_cached.cache_clear()

def build_networkx_graph(env: Any) -> nx.DiGraph:
    """Build NetworkX directed graph from environment (cached per run only)."""
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache
    
    G = nx.DiGraph()
    road_data = env.get_road_network_data()
    
    # Add weighted edges from adjacency list
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
    """
    Cached shortest path using NetworkX.
    run_id ensures cache is unique per solver() call.
    Returns (path_tuple, distance).
    """
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
    
    # Use run counter to ensure cache is per-run only
    path_tuple, distance = get_shortest_path_cached(start, end, _cache_counter)
    
    if path_tuple is None:
        return None, float('inf')
    
    return list(path_tuple), distance

def build_steps_with_path(env: Any, wh_node: int, delivery_nodes: List[int], order_assignments: Dict) -> List[Dict]:
    """Build route steps with proper pathfinding."""
    steps = [{"node_id": wh_node, "pickups": [], "deliveries": [], "unloads": []}]
    
    current = wh_node
    for delivery_node in delivery_nodes:
        path, _ = dijkstra_shortest_path(env, current, delivery_node)
        if path is None:
            path = [current, delivery_node]
        
        # Add intermediate nodes
        for i, node in enumerate(path):
            if node == current and i == 0:
                continue
            
            if node == delivery_node:
                deliveries = order_assignments.get(delivery_node, [])
                steps.append({"node_id": node, "pickups": [], "deliveries": deliveries, "unloads": []})
            else:
                steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
        
        current = delivery_node
    
    # Return to warehouse
    path_back, _ = dijkstra_shortest_path(env, current, wh_node)
    if path_back and len(path_back) > 1:
        for node in path_back[1:]:
            if node == wh_node:
                steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
            else:
                steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
    
    return steps

def calculate_route_distance(env: Any, steps: List[Dict]) -> float:
    """Calculate total distance for a route."""
    total_distance = 0.0
    for i in range(len(steps) - 1):
        node1 = steps[i]["node_id"]
        node2 = steps[i + 1]["node_id"]
        _, dist = dijkstra_shortest_path(env, node1, node2)
        if dist != float('inf'):
            total_distance += dist
    return total_distance

def two_opt_optimize_route(env: Any, route: Dict, max_time: float = 5.0) -> Dict:
    """
    2-opt local search to optimize route order.
    Swaps delivery order to minimize total distance.
    """
    start_time = time.time()
    steps = route.get("steps", [])
    
    if len(steps) <= 4:  # Too short to optimize
        return route
    
    # Find indices of delivery stops (skip warehouse pickup/return)
    delivery_indices = []
    for i, step in enumerate(steps):
        if step.get("deliveries"):
            delivery_indices.append(i)
    
    if len(delivery_indices) <= 2:
        return route
    
    # Current best
    best_steps = list(steps)
    best_distance = calculate_route_distance(env, best_steps)
    
    improved = True
    iterations = 0
    
    while improved and time.time() - start_time < max_time:
        improved = False
        iterations += 1
        
        for i in range(len(delivery_indices) - 1):
            for j in range(i + 2, len(delivery_indices)):
                if time.time() - start_time >= max_time:
                    break
                
                # Create new route with reversed segment
                new_steps = list(best_steps)
                idx_i = delivery_indices[i]
                idx_j = delivery_indices[j]
                
                # Reverse the segment between i and j
                new_steps[idx_i:idx_j+1] = reversed(new_steps[idx_i:idx_j+1])
                
                # Calculate new distance
                new_distance = calculate_route_distance(env, new_steps)
                
                if new_distance < best_distance - 0.01:
                    best_distance = new_distance
                    best_steps = new_steps
                    improved = True
            
            if time.time() - start_time >= max_time:
                break
    
    return {"vehicle_id": route["vehicle_id"], "steps": best_steps}

def solver(env: Any) -> Dict:
    """
    Adaptive high-performance solver.
    
    STRATEGY:
    1. Clear all caches (no cross-run persistence)
    2. Smart vehicle ranking by cost-per-capacity
    3. Adaptive bin packing with large vehicles first
    4. Works across ALL scenarios (not hardcoded for specific case)
    """
    # CRITICAL: Clear all caches at start of each run
    clear_caches()
    
    print("\n[SOLVER 53] Adaptive High-Performance Solver")
    print("=" * 80)
    
    # Build graph once at start
    G = build_networkx_graph(env)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Get all vehicles and calculate cost-per-capacity
    all_vehicles = list(env.get_all_vehicles())
    
    # Calculate cost efficiency for each vehicle type
    vehicle_efficiency = {}
    for v in all_vehicles:
        # Cost per unit capacity (lower is better)
        capacity_units = v.capacity_weight + v.capacity_volume
        cost_per_unit = v.fixed_cost / capacity_units if capacity_units > 0 else float('inf')
        
        if v.type not in vehicle_efficiency:
            vehicle_efficiency[v.type] = {
                'cost_per_unit': cost_per_unit,
                'fixed_cost': v.fixed_cost,
                'capacity_weight': v.capacity_weight,
                'capacity_volume': v.capacity_volume
            }
    
    print(f"\nVehicle efficiency (cost per unit capacity):")
    for vtype, info in sorted(vehicle_efficiency.items(), key=lambda x: x[1]['cost_per_unit']):
        print(f"  {vtype}: ${info['cost_per_unit']:.3f}/unit (${info['fixed_cost']} fixed, {info['capacity_weight']}kg/{info['capacity_volume']}m³)")
    
    # Sort vehicle types by efficiency (best first)
    sorted_types = sorted(vehicle_efficiency.keys(), key=lambda x: vehicle_efficiency[x]['cost_per_unit'])
    
    # Group vehicles by warehouse and type
    vehicles_by_wh = {}
    for v in all_vehicles:
        wh_id = v.home_warehouse_id
        if wh_id not in vehicles_by_wh:
            vehicles_by_wh[wh_id] = {vtype: [] for vtype in sorted_types}
        
        if v.type not in vehicles_by_wh[wh_id]:
            vehicles_by_wh[wh_id][v.type] = []
        vehicles_by_wh[wh_id][v.type].append(v)
    
    # Get all orders
    all_orders = list(env.get_all_order_ids())
    print(f"\nTotal orders: {len(all_orders)}")
    
    # Adaptive warehouse allocation: Balance by total weight/volume
    order_data = []
    for oid in all_orders:
        req = env.get_order_requirements(oid)
        total_weight = sum(env.skus[sku].weight * qty for sku, qty in req.items())
        total_volume = sum(env.skus[sku].volume * qty for sku, qty in req.items())
        order_data.append((oid, total_weight, total_volume))
    
    # Sort by size (large first) for efficient packing
    order_data.sort(key=lambda x: max(x[1], x[2]), reverse=True)
    
    # Allocate to warehouses alternating (balances load)
    warehouse_ids = list(env.warehouses.keys())
    orders_by_wh = {wh_id: [] for wh_id in warehouse_ids}
    
    for i, (oid, w, v) in enumerate(order_data):
        wh_id = warehouse_ids[i % len(warehouse_ids)]
        orders_by_wh[wh_id].append(oid)
    
    for wh_id, orders in orders_by_wh.items():
        print(f"  {wh_id}: {len(orders)} orders")
    
    # BUILD ROUTES
    routes = []
    
    for wh_id, orders in orders_by_wh.items():
        if not orders:
            continue
            
        wh = env.warehouses[wh_id]
        wh_node = wh.location.id
        
        # Get vehicles for this warehouse, sorted by efficiency
        available_vehicles = []
        for vtype in sorted_types:
            if wh_id in vehicles_by_wh and vtype in vehicles_by_wh[wh_id]:
                available_vehicles.extend(vehicles_by_wh[wh_id][vtype])
        
        print(f"\n[{wh_id}] Packing {len(orders)} orders")
        
        # ADAPTIVE BIN PACKING
        truck_loads = []
        remaining_orders = list(orders)
        
        # Sort orders by size (large first) for efficient packing
        order_data = []
        for oid in remaining_orders:
            req = env.get_order_requirements(oid)
            weight = sum(env.skus[sku].weight * qty for sku, qty in req.items())
            volume = sum(env.skus[sku].volume * qty for sku, qty in req.items())
            order_data.append((oid, weight, volume))
        
        order_data.sort(key=lambda x: max(x[1], x[2]), reverse=True)
        remaining_orders = [o[0] for o in order_data]
        
        # Pack into vehicles in efficiency order
        for truck in available_vehicles:
            if not remaining_orders:
                break
            
            packed_orders = []
            rem_weight = truck.capacity_weight
            rem_volume = truck.capacity_volume
            
            # Greedy pack
            for oid in remaining_orders[:]:
                req = env.get_order_requirements(oid)
                order_weight = sum(env.skus[sku].weight * qty for sku, qty in req.items())
                order_volume = sum(env.skus[sku].volume * qty for sku, qty in req.items())
                
                if order_weight <= rem_weight and order_volume <= rem_volume:
                    packed_orders.append(oid)
                    remaining_orders.remove(oid)
                    rem_weight -= order_weight
                    rem_volume -= order_volume
            
            if packed_orders:
                util_w = (truck.capacity_weight - rem_weight) / truck.capacity_weight * 100
                util_v = (truck.capacity_volume - rem_volume) / truck.capacity_volume * 100
                print(f"  {truck.type}: {len(packed_orders)} orders, {util_w:.1f}%W {util_v:.1f}%V")
                truck_loads.append((truck, packed_orders))
        
        if remaining_orders:
            print(f"  WARNING: {len(remaining_orders)} orders could not be packed!")
        
        # BUILD ROUTES for each truck
        for truck, packed_orders in truck_loads:
            if not packed_orders:
                continue
            
            # Get order destinations
            delivery_nodes = []
            order_assignments = {}
            
            for oid in packed_orders:
                order = env.orders[oid]
                dest_node = order.destination.id
                
                if dest_node not in order_assignments:
                    order_assignments[dest_node] = []
                    delivery_nodes.append(dest_node)
                
                req = env.get_order_requirements(oid)
                for sku_id, qty in req.items():
                    order_assignments[dest_node].append({
                        "order_id": oid,
                        "sku_id": sku_id,
                        "quantity": qty
                    })
            
            # Simple nearest-neighbor TSP
            if len(delivery_nodes) > 1:
                unvisited = set(delivery_nodes)
                tsp_order = []
                current = wh_node
                
                while unvisited:
                    nearest = min(unvisited, key=lambda n: dijkstra_shortest_path(env, current, n)[1])
                    tsp_order.append(nearest)
                    unvisited.remove(nearest)
                    current = nearest
                
                delivery_nodes = tsp_order
            
            # Build route with pickups
            steps = [{"node_id": wh_node, "pickups": [], "deliveries": [], "unloads": []}]
            
            # Add pickups at warehouse
            for oid in packed_orders:
                req = env.get_order_requirements(oid)
                for sku_id, qty in req.items():
                    steps[0]["pickups"].append({
                        "warehouse_id": wh_id,
                        "sku_id": sku_id,
                        "quantity": qty
                    })
            
            # Add delivery steps with pathfinding
            current = wh_node
            for delivery_node in delivery_nodes:
                path, _ = dijkstra_shortest_path(env, current, delivery_node)
                if path is None:
                    path = [current, delivery_node]
                
                for i, node in enumerate(path):
                    if node == current and i == 0:
                        continue
                    
                    if node == delivery_node:
                        deliveries = order_assignments.get(delivery_node, [])
                        steps.append({"node_id": node, "pickups": [], "deliveries": deliveries, "unloads": []})
                    else:
                        steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
                
                current = delivery_node
            
            # Return to warehouse
            path_back, _ = dijkstra_shortest_path(env, current, wh_node)
            if path_back and len(path_back) > 1:
                for node in path_back[1:]:
                    if node == wh_node:
                        steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
                    else:
                        steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
            
            route = {"vehicle_id": truck.id, "steps": steps}
            routes.append(route)
    
    # OPTIMIZATION: Apply 2-opt to reduce route distances
    print(f"\n[OPTIMIZATION] Applying 2-opt to {len(routes)} routes...")
    optimized_routes = []
    for i, route in enumerate(routes):
        optimized = two_opt_optimize_route(env, route, max_time=3.0)
        optimized_routes.append(optimized)
        print(f"  Route {i+1}: Optimized")
    
    print(f"\n[SOLUTION] Generated {len(optimized_routes)} optimized routes")
    return {"routes": optimized_routes}


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
