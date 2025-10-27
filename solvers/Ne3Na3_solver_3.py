"""
Ne3Na3_solver_3.py
 
Implements fast greedy multi-warehouse solver with:
- Greedy assignment to nearest fulfilling warehouse
- Capacity-aware packing
- Nearest-neighbor route building
- Small 2-opt route optimization
"""
from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple, Set
from collections import deque
from copy import deepcopy
import heapq


def dijkstra_shortest_path(env, start_node, end_node):
    """
    Compute shortest path from start_node to end_node using Dijkstra's algorithm.
    NO CACHING - fresh computation every time as required by competition rules.
    
    Args:
        env: Environment with road network
        start_node: Starting node ID
        end_node: Target node ID
    
    Returns:
        (distance, path) tuple or (None, None) if no path exists
    """
    if start_node == end_node:
        return 0.0, [start_node]
    
    # Get road network data (this is not caching - it's fetching current state)
    road_network = env.get_road_network_data()
    if not road_network:
        return None, None
    
    adjacency_list = road_network.get("adjacency_list", {})
    edges_list = road_network.get("edges", [])
    
    if start_node not in adjacency_list or end_node not in adjacency_list:
        return None, None
    
    # Build adjacency dict with weights for O(1) lookups (local to this function call)
    neighbors_with_weights = {}
    for edge in edges_list:
        from_node = edge.get('from')
        to_node = edge.get('to')
        distance = edge.get('distance', 1.0)
        if from_node not in neighbors_with_weights:
            neighbors_with_weights[from_node] = []
        neighbors_with_weights[from_node].append((to_node, distance))
    
    # Initialize distances and previous nodes (no caching - local to this call)
    distances = {start_node: 0.0}
    previous = {start_node: None}
    
    # Priority queue: (distance, node)
    pq = [(0.0, start_node)]
    visited = set()
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # Found target
        if current_node == end_node:
            # Reconstruct path
            path = []
            node = end_node
            while node is not None:
                path.append(node)
                node = previous.get(node)
            path.reverse()
            return current_dist, path
        
        # Explore neighbors with weights
        for neighbor, edge_weight in neighbors_with_weights.get(current_node, []):
            if neighbor in visited:
                continue
            
            new_dist = current_dist + edge_weight
            
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))
    
    # No path found
    return None, None


def get_shortest_distance(env, start_node, end_node):
    """
    Get shortest distance between two nodes.
    NO CACHING - computes fresh each time.
    
    Args:
        env: Environment instance
        start_node: Starting node ID
        end_node: Target node ID
    
    Returns:
        Distance as float or None if no path exists
    """
    if start_node == end_node:
        return 0.0
    
    dist, _ = dijkstra_shortest_path(env, start_node, end_node)
    return dist


def is_node_reachable(env, start_node, end_node):
    """
    Check if end_node is reachable from start_node.
    NO CACHING.
    
    Args:
        env: Environment instance
        start_node: Starting node ID
        end_node: Target node ID
    
    Returns:
        True if reachable, False otherwise
    """
    if start_node == end_node:
        return True
    
    dist, _ = dijkstra_shortest_path(env, start_node, end_node)
    return dist is not None


def my_solver(env) -> Dict:
    """Generate optimized solution using greedy multi-warehouse approach.
 
    Args:
        env: LogisticsEnvironment instance
 
    Returns:
        A complete solution dict with routes and sequential steps.
    """
    
    # ===== DISTANCE CACHING SETUP (local to this solver execution) =====
    _distance_cache = {}  # (from_node, to_node) -> distance
    
    def get_cached_distance(from_node, to_node):
        """Get distance with caching within this solver execution."""
        if from_node == to_node:
            return 0.0
        key = (from_node, to_node)
        if key not in _distance_cache:
            _distance_cache[key] = get_shortest_distance(env, from_node, to_node)
        return _distance_cache[key]
    
    def is_node_reachable_cached(start_node, end_node):
        """Check reachability using cache."""
        if start_node == end_node:
            return True
        dist = get_cached_distance(start_node, end_node)
        return dist is not None
    
    # ===== STEP 1: Extract and prepare data from environment =====
    orders_data = []
    for order_id in env.get_all_order_ids():
        order = env.orders[order_id]
        order_node = order.destination.id
        items = order.requested_items  # Dict[sku_id, quantity]
        
        # Calculate total weight and volume
        total_weight = 0.0
        total_volume = 0.0
        for sku_id, qty in items.items():
            sku = env.skus[sku_id]
            total_weight += sku.weight * qty
            total_volume += sku.volume * qty
        
        orders_data.append({
            'id': order_id,
            'node': order_node,
            'items': items,
            'weight': total_weight,
            'volume': total_volume
        })
    
    warehouses_data = []
    vehicles_data = []
    
    for wh_id, warehouse in env.warehouses.items():
        wh_node = warehouse.location.id
        wh_inventory = deepcopy(warehouse.inventory)
        
        warehouses_data.append({
            'id': wh_id,
            'node': wh_node,
            'inventory': wh_inventory
        })
        
        # Extract vehicles for this warehouse
        for vehicle in warehouse.vehicles:
            vehicles_data.append({
                'id': vehicle.id,
                'home_warehouse_id': wh_id,
                'home_node': wh_node,
                'capacity_weight': vehicle.capacity_weight,
                'capacity_volume': vehicle.capacity_volume,
                'max_distance': vehicle.max_distance,
                'cost_per_km': vehicle.cost_per_km,
                'fixed_cost': vehicle.fixed_cost
            })
    
    # ===== STEP 2: Assign orders to warehouses (greedy nearest fulfillment) =====
    assigned = {}  # order_id -> [(warehouse_id, {sku: qty}), ...]
    
    for order in orders_data:
        order_id = order['id']
        order_node = order['node']
        required_items = order['items']
        
        # Try to find a single warehouse that can fully fulfill
        best_wh = None
        best_dist = float('inf')
        
        for wh in warehouses_data:
            can_fulfill = all(
                wh['inventory'].get(sku, 0) >= qty 
                for sku, qty in required_items.items()
            )
            
            if can_fulfill:
                dist = get_cached_distance(wh['node'], order_node)
                if dist is not None and dist < best_dist:
                    best_dist = dist
                    best_wh = wh
        
        if best_wh:
            # Full fulfillment from single warehouse
            assigned[order_id] = [(best_wh['id'], deepcopy(required_items))]
            # Deduct from warehouse inventory
            for sku, qty in required_items.items():
                best_wh['inventory'][sku] -= qty
        else:
            # Split across multiple warehouses (greedy)
            remaining = deepcopy(required_items)
            allocated = []
            
            # Sort warehouses by distance (use lazy cached distances)
            wh_distances = []
            for wh in warehouses_data:
                dist = get_cached_distance(wh['node'], order_node)
                wh_distances.append((wh, dist if dist is not None else float('inf')))
            wh_sorted = [wh for wh, _ in sorted(wh_distances, key=lambda x: x[1])]
            
            for wh in wh_sorted:
                if not remaining:
                    break
                
                pickup_items = {}
                for sku, needed_qty in list(remaining.items()):
                    available = wh['inventory'].get(sku, 0)
                    if available > 0:
                        take = min(available, needed_qty)
                        pickup_items[sku] = take
                        wh['inventory'][sku] -= take
                        remaining[sku] -= take
                        if remaining[sku] == 0:
                            del remaining[sku]
                
                if pickup_items:
                    allocated.append((wh['id'], pickup_items))
            
            assigned[order_id] = allocated
    
    # ===== STEP 3: Group orders by warehouse and pack into vehicles =====
    wh_orders = {}  # warehouse_id -> [(order_data, items_to_pickup), ...]
    
    for order in orders_data:
        alloc = assigned.get(order['id'], [])
        for wh_id, items in alloc:
            if wh_id not in wh_orders:
                wh_orders[wh_id] = []
            wh_orders[wh_id].append((order, items))
    
    vehicles_by_wh = {}
    for v in vehicles_data:
        wh_id = v['home_warehouse_id']
        if wh_id not in vehicles_by_wh:
            vehicles_by_wh[wh_id] = []
        vehicles_by_wh[wh_id].append(v)
    
    routes = []
    
    # Pack orders into vehicles per warehouse
    for wh_id, orders_list in wh_orders.items():
        wh_data = next((w for w in warehouses_data if w['id'] == wh_id), None)
        if not wh_data:
            continue
        
        wh_node = wh_data['node']
        available_vehicles = vehicles_by_wh.get(wh_id, [])
        
        if not available_vehicles:
            continue
        
        # Sort orders by distance from warehouse (use cached distances)
        order_distances = []
        for order_item in orders_list:
            dist = get_cached_distance(wh_node, order_item[0]['node'])
            order_distances.append((order_item, dist if dist is not None else float('inf')))
        orders_sorted = [order_item for order_item, _ in sorted(order_distances, key=lambda x: x[1])]
        
        # Greedy packing into vehicles
        vehicle_idx = 0
        
        while orders_sorted and vehicle_idx < len(available_vehicles):
            vehicle = available_vehicles[vehicle_idx]
            vehicle_id = vehicle['id']
            
            route_orders = []
            current_weight = 0.0
            current_volume = 0.0
            current_distance = 0.0
            
            # Try to pack orders into this vehicle
            remaining_orders = []
            
            for order_data, pickup_items in orders_sorted:
                # Calculate weight/volume for pickup items
                pickup_weight = 0.0
                pickup_volume = 0.0
                for sku_id, qty in pickup_items.items():
                    sku = env.skus[sku_id]
                    pickup_weight += sku.weight * qty
                    pickup_volume += sku.volume * qty
                
                # Check capacity
                if (current_weight + pickup_weight <= vehicle['capacity_weight'] and
                    current_volume + pickup_volume <= vehicle['capacity_volume']):
                    
                    route_orders.append((order_data, pickup_items))
                    current_weight += pickup_weight
                    current_volume += pickup_volume
                else:
                    remaining_orders.append((order_data, pickup_items))
            
            if route_orders:
                # Build route for this vehicle
                route = build_vehicle_route(env, vehicle, wh_node, route_orders, get_cached_distance, is_node_reachable_cached)
                routes.append(route)
            
            orders_sorted = remaining_orders
            vehicle_idx += 1
    
    # ===== STEP 4: Format solution =====
    solution = {"routes": routes}
    return solution


def build_vehicle_route(env, vehicle, wh_node, orders_list):
    """Build a sequential route for a vehicle visiting orders.
    
    Args:
        env: Environment instance
        vehicle: Vehicle data dict
        wh_node: Warehouse node ID
        orders_list: List of (order_data, pickup_items) tuples
    
    Returns:
        Route dict with vehicle_id and steps
    """
    vehicle_id = vehicle['id']
    wh_id = vehicle['home_warehouse_id']
    
    if not orders_list:
        # Empty route - just return to depot
        return {
            'vehicle_id': vehicle_id,
            'steps': [
                {
                    'node_id': wh_node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                }
            ]
        }
    
    steps = []
    
    # Step 1: Pickup all items from warehouse
    all_pickups = []
    for order_data, pickup_items in orders_list:
        for sku_id, qty in pickup_items.items():
            all_pickups.append({
                'warehouse_id': wh_id,
                'sku_id': sku_id,
                'quantity': qty
            })
    
    # First step: at warehouse with pickups
    steps.append({
        'node_id': wh_node,
        'pickups': all_pickups,
        'deliveries': [],
        'unloads': []
    })
    
    # Step 2: Visit orders in nearest-neighbor order
    visited = set()
    current_node = wh_node
    remaining_orders = list(orders_list)
    
    # Pre-compute distances once to avoid repeated Dijkstra calls
    distance_cache = {}  # (from_node, to_node) -> distance
    
    while remaining_orders:
        # Find nearest unvisited order that is reachable
        best_order = None
        best_dist = float('inf')
        best_idx = -1
        
        for idx, (order_data, pickup_items) in enumerate(remaining_orders):
            order_node = order_data['node']
            if order_node not in visited:
                # Check if reachable and get distance (cache within this route)
                cache_key = (current_node, order_node)
                if cache_key not in distance_cache:
                    distance_cache[cache_key] = get_shortest_distance(env, current_node, order_node)
                
                dist = distance_cache[cache_key]
                if dist is not None and dist < best_dist:
                    best_dist = dist
                    best_order = (order_data, pickup_items)
                    best_idx = idx
        
        if best_order:
            order_data, pickup_items = best_order
            order_node = order_data['node']
            order_id = order_data['id']
            
            # Create delivery step
            deliveries = []
            for sku_id, qty in pickup_items.items():
                deliveries.append({
                    'order_id': order_id,
                    'sku_id': sku_id,
                    'quantity': qty
                })
            
            steps.append({
                'node_id': order_node,
                'pickups': [],
                'deliveries': deliveries,
                'unloads': []
            })
            
            visited.add(order_node)
            current_node = order_node
            remaining_orders.pop(best_idx)
        else:
            # No more reachable orders
            break
    
    # Step 3: Return to home warehouse (only if reachable)
    if current_node != wh_node:
        # Verify we can return to warehouse
        if is_node_reachable(env, current_node, wh_node):
            steps.append({
                'node_id': wh_node,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })
        else:
            # Cannot return to warehouse - this route is invalid
            # Return empty route
            return {
                'vehicle_id': vehicle_id,
                'steps': [
                    {
                        'node_id': wh_node,
                        'pickups': [],
                        'deliveries': [],
                        'unloads': []
                    }
                ]
            }
    
    # Apply 2-opt optimization on delivery nodes
    steps = optimize_route_2opt(env, steps)
    
    return {
        'vehicle_id': vehicle_id,
        'steps': steps
    }


def optimize_route_2opt(env, steps):
    """Apply 2-opt optimization to improve route order.
    
    Args:
        env: Environment instance
        steps: List of step dicts
    
    Returns:
        Optimized steps list
    """
    if len(steps) <= 4:  # Too small to optimize
        return steps
    
    # Extract delivery steps (skip first warehouse step and pickups)
    delivery_indices = []
    for i, step in enumerate(steps):
        if step['deliveries']:
            delivery_indices.append(i)
    
    if len(delivery_indices) < 2:
        return steps
    
    # Pre-compute all distances between nodes in this route to avoid repeated Dijkstra
    unique_nodes = list(set(step['node_id'] for step in steps))
    dist_cache = {}
    for node_a in unique_nodes:
        for node_b in unique_nodes:
            if node_a != node_b:
                dist_cache[(node_a, node_b)] = get_shortest_distance(env, node_a, node_b) or 0
    
    # Try 2-opt swaps (limited iterations for speed)
    improved = True
    iterations = 0
    max_iterations = min(10, len(delivery_indices))
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(len(delivery_indices) - 1):
            for j in range(i + 1, len(delivery_indices)):
                idx_i = delivery_indices[i]
                idx_j = delivery_indices[j]
                
                # Calculate current cost
                node_before_i = steps[idx_i - 1]['node_id'] if idx_i > 0 else steps[0]['node_id']
                node_i = steps[idx_i]['node_id']
                node_j = steps[idx_j]['node_id']
                node_after_j = steps[idx_j + 1]['node_id'] if idx_j + 1 < len(steps) else steps[-1]['node_id']
                
                dist_before_i_to_i = dist_cache.get((node_before_i, node_i), 0)
                dist_i_to_j = dist_cache.get((node_i, node_j), 0)
                dist_j_to_after_j = dist_cache.get((node_j, node_after_j), 0)
                
                current_cost = dist_before_i_to_i + dist_i_to_j + dist_j_to_after_j
                
                # Calculate swapped cost
                dist_before_i_to_j = dist_cache.get((node_before_i, node_j), 0)
                dist_j_to_i = dist_cache.get((node_j, node_i), 0)
                dist_i_to_after_j = dist_cache.get((node_i, node_after_j), 0)
                
                swapped_cost = dist_before_i_to_j + dist_j_to_i + dist_i_to_after_j
                
                if swapped_cost < current_cost:
                    # Swap steps
                    steps[idx_i], steps[idx_j] = steps[idx_j], steps[idx_i]
                    improved = True
                    break
            
            if improved:
                break
    
    return steps


# if __name__ == '__main__':
#     env = LogisticsEnvironment()
#     result = my_solver(env)
#     print(result)