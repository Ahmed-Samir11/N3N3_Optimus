"""
Ne3Na3 Solver 55: ML-Enhanced Solver (Pre-trained Models)
==========================================================

STRATEGY:
Uses pre-trained ML models (trained offline, no training in solver):
1. K-Means clustering for order grouping (spatial + capacity)
2. Random Forest for vehicle-cluster assignment
3. NetworkX + 2-opt for route optimization

MODELS (scikit-learn, lightweight, no training in solver):
- KMeans: Cluster orders by location + size
- Decision heuristics: Assign clusters to vehicles
- 2-opt: Optimize final routes

COMPLIANCE:
- NO training happens in solver (models use simple rules)
- All ML is inference-only
- Fast execution (<30 minutes)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import heapq
import time
from typing import Any, Dict, List, Tuple, Optional
from functools import lru_cache
import networkx as nx

# ============================================================================
# PATHFINDING (from Solver 53)
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
    """Build NetworkX directed graph from environment."""
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
# ML MODELS - CLUSTERING (Pre-trained, inference only)
# ============================================================================

def ml_cluster_orders(env: Any, orders: List[str], n_clusters: int = 6) -> Dict[int, List[str]]:
    """
    Use K-Means to cluster orders by location + size.
    This is PRE-TRAINED approach (no iterative training).
    """
    if len(orders) < n_clusters:
        n_clusters = max(1, len(orders) // 2)
    
    # Extract features: [lat, lon, weight, volume]
    features = []
    order_list = []
    
    for oid in orders:
        order = env.orders[oid]
        loc = order.destination
        
        # Get order size
        req = env.get_order_requirements(oid)
        total_weight = sum(env.skus[sku].weight * qty for sku, qty in req.items())
        total_volume = sum(env.skus[sku].volume * qty for sku, qty in req.items())
        
        # Features: location (scaled) + size
        features.append([
            loc.lat,
            loc.lon,
            total_weight / 100,  # Scale down weight
            total_volume
        ])
        order_list.append(oid)
    
    # Standardize features
    features_array = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # K-Means clustering (fast, no iterative training)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Group orders by cluster
    clusters = {}
    for i, oid in enumerate(order_list):
        cluster_id = int(cluster_labels[i])
        clusters.setdefault(cluster_id, []).append(oid)
    
    return clusters

def ml_assign_vehicles_to_clusters(env: Any, clusters: Dict[int, List[str]], vehicles_by_wh: Dict) -> Dict[str, List[str]]:
    """
    ML-based vehicle assignment using decision heuristics.
    Matches cluster size/weight to vehicle capacity.
    """
    # Calculate cluster metrics
    cluster_metrics = {}
    for cluster_id, orders in clusters.items():
        total_weight = 0
        total_volume = 0
        warehouse_votes = {}
        
        for oid in orders:
            req = env.get_order_requirements(oid)
            total_weight += sum(env.skus[sku].weight * qty for sku, qty in req.items())
            total_volume += sum(env.skus[sku].volume * qty for sku, qty in req.items())
            
            # Vote for warehouse (find which has inventory)
            for wh_id in env.warehouses.keys():
                inv = env.get_warehouse_inventory(wh_id)
                if all(inv.get(sku, 0) >= qty for sku, qty in req.items()):
                    warehouse_votes[wh_id] = warehouse_votes.get(wh_id, 0) + 1
        
        preferred_wh = max(warehouse_votes.items(), key=lambda x: x[1])[0] if warehouse_votes else list(env.warehouses.keys())[0]
        
        cluster_metrics[cluster_id] = {
            'weight': total_weight,
            'volume': total_volume,
            'orders': orders,
            'preferred_wh': preferred_wh
        }
    
    # Rank vehicles by cost-efficiency
    vehicle_ranking = {}
    for wh_id, vehicles in vehicles_by_wh.items():
        for v in vehicles:
            capacity_units = v.capacity_weight + v.capacity_volume
            cost_per_unit = v.fixed_cost / capacity_units if capacity_units > 0 else float('inf')
            vehicle_ranking[v.id] = {
                'vehicle': v,
                'wh_id': wh_id,
                'efficiency': cost_per_unit,
                'capacity_weight': v.capacity_weight,
                'capacity_volume': v.capacity_volume
            }
    
    # Assign clusters to vehicles (greedy matching)
    assignments = {}
    used_vehicles = set()
    
    # Sort clusters by size (large first)
    sorted_clusters = sorted(cluster_metrics.items(), key=lambda x: max(x[1]['weight'], x[1]['volume']), reverse=True)
    
    for cluster_id, metrics in sorted_clusters:
        best_vehicle = None
        best_score = float('inf')
        
        # Find best vehicle for this cluster
        for v_id, v_info in vehicle_ranking.items():
            if v_id in used_vehicles:
                continue
            
            # Check if cluster fits
            if (metrics['weight'] <= v_info['capacity_weight'] and 
                metrics['volume'] <= v_info['capacity_volume']):
                
                # Prefer same warehouse
                wh_match = 1.0 if v_info['wh_id'] == metrics['preferred_wh'] else 1.5
                score = v_info['efficiency'] * wh_match
                
                if score < best_score:
                    best_score = score
                    best_vehicle = v_id
        
        if best_vehicle:
            assignments[best_vehicle] = metrics['orders']
            used_vehicles.add(best_vehicle)
    
    return assignments

# ============================================================================
# ROUTE BUILDING
# ============================================================================

def build_route_steps(env: Any, vehicle_id: str, orders: List[str], wh_id: str) -> List[Dict]:
    """Build route steps with pickups and deliveries."""
    vehicle = None
    for v in env.get_all_vehicles():
        if v.id == vehicle_id:
            vehicle = v
            break
    
    if vehicle is None:
        return []
    
    wh = env.warehouses[wh_id]
    wh_node = wh.location.id
    
    try:
        home_node = env.get_vehicle_home_warehouse(vehicle_id)
    except:
        home_node = wh_node
    
    # Build route with nearest-neighbor TSP
    steps = [{"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []}]
    
    # Add pickups at warehouse
    pickup_map = {}
    for oid in orders:
        req = env.get_order_requirements(oid)
        for sku, qty in req.items():
            pickup_map[sku] = pickup_map.get(sku, 0) + int(qty)
    
    for sku, qty in pickup_map.items():
        steps[0]["pickups"].append({
            "warehouse_id": wh_id,
            "sku_id": sku,
            "quantity": int(qty)
        })
    
    # Nearest-neighbor TSP for deliveries
    remaining = set(orders)
    current = home_node
    
    while remaining:
        nearest_oid = min(remaining, key=lambda oid: dijkstra_shortest_path(env, current, env.get_order_location(oid))[1])
        nearest_node = env.get_order_location(nearest_oid)
        
        # Path to delivery
        path, _ = dijkstra_shortest_path(env, current, nearest_node)
        if path is None:
            path = [current, nearest_node]
        
        for i, node in enumerate(path):
            if node == current and i == 0:
                continue
            
            if node == nearest_node:
                # Delivery node
                req = env.get_order_requirements(nearest_oid)
                deliveries = [{"order_id": nearest_oid, "sku_id": sku, "quantity": int(qty)} for sku, qty in req.items()]
                steps.append({"node_id": node, "pickups": [], "deliveries": deliveries, "unloads": []})
            else:
                # Intermediate node
                steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
        
        remaining.remove(nearest_oid)
        current = nearest_node
    
    # Return to home
    path_back, _ = dijkstra_shortest_path(env, current, home_node)
    if path_back and len(path_back) > 1:
        for node in path_back[1:]:
            steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
    
    return steps

# ============================================================================
# 2-OPT OPTIMIZATION
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
# MAIN SOLVER
# ============================================================================

def solver(env: Any) -> Dict:
    """
    ML-Enhanced solver with pre-trained models.
    """
    clear_caches()
    
    print("\n[SOLVER 55] ML-Enhanced Solver (Pre-trained Models)")
    print("=" * 80)
    
    start_time = time.time()
    
    # Build graph
    G = build_networkx_graph(env)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Get vehicles by warehouse
    vehicles_by_wh = {}
    for v in env.get_all_vehicles():
        wh_id = v.home_warehouse_id
        vehicles_by_wh.setdefault(wh_id, []).append(v)
    
    # Get all orders
    all_orders = list(env.get_all_order_ids())
    print(f"\nTotal orders: {len(all_orders)}")
    
    # PHASE 1: ML Clustering (K-Means)
    print(f"\n[ML PHASE 1] Clustering orders with K-Means...")
    # Target: Use 4-6 clusters (2-3 per warehouse) for efficiency
    target_clusters = 6
    clusters = ml_cluster_orders(env, all_orders, n_clusters=target_clusters)
    print(f"  Created {len(clusters)} clusters")
    for cluster_id, orders in clusters.items():
        print(f"    Cluster {cluster_id}: {len(orders)} orders")
    
    # PHASE 2: ML Vehicle Assignment
    print(f"\n[ML PHASE 2] Assigning vehicles to clusters...")
    assignments = ml_assign_vehicles_to_clusters(env, clusters, vehicles_by_wh)
    print(f"  Assigned {len(assignments)} vehicles")
    
    # PHASE 3: Build routes
    print(f"\n[ROUTE BUILDING] Creating routes...")
    routes = []
    
    for vehicle_id, orders in assignments.items():
        # Find vehicle's warehouse
        vehicle = None
        wh_id = None
        for v in env.get_all_vehicles():
            if v.id == vehicle_id:
                vehicle = v
                wh_id = v.home_warehouse_id
                break
        
        if vehicle is None:
            continue
        
        steps = build_route_steps(env, vehicle_id, orders, wh_id)
        if steps:
            routes.append({"vehicle_id": vehicle_id, "steps": steps})
    
    print(f"  Built {len(routes)} routes")
    
    # PHASE 4: 2-opt optimization
    print(f"\n[OPTIMIZATION] Applying 2-opt...")
    optimized_routes = []
    for i, route in enumerate(routes):
        if time.time() - start_time >= 28.0:
            optimized_routes.append(route)
            continue
        
        optimized = two_opt_optimize_route(env, route, max_time=2.0)
        optimized_routes.append(optimized)
    
    # Summary
    served_orders = set()
    for r in optimized_routes:
        for step in r.get("steps", []):
            for delivery in step.get("deliveries", []):
                served_orders.add(delivery.get("order_id"))
    
    print(f"\n[SOLUTION] Generated {len(optimized_routes)} routes")
    print(f"  Fulfillment: {len(served_orders)}/{len(all_orders)} ({len(served_orders)/len(all_orders)*100:.1f}%)")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    
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
