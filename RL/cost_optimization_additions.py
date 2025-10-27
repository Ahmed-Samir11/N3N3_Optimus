"""
Cost optimization improvements for solver 84

Add these functions to your solver to reduce costs:
"""

def two_opt_route_optimization(delivery_nodes, env, distance_cache):
    """
    2-opt local search: iteratively improve route by swapping edges
    
    Example:
    Route: A -> B -> C -> D -> E
    Try swapping edge (B,C)-(D,E) to (B,D)-(C,E)
    If shorter, keep the swap
    
    Returns: Optimized order of delivery nodes
    """
    if len(delivery_nodes) <= 2:
        return delivery_nodes
    
    route = delivery_nodes.copy()
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                # Current edges: (route[i], route[i+1]) and (route[j-1], route[j])
                # New edges: (route[i], route[j-1]) and (route[i+1], route[j])
                
                # Calculate current distance
                current_dist = 0
                if i + 1 < len(route):
                    d, _ = dijkstra_shortest_path(env, route[i], route[i+1], distance_cache)
                    current_dist += d if d else float('inf')
                if j < len(route):
                    d, _ = dijkstra_shortest_path(env, route[j-1], route[j], distance_cache)
                    current_dist += d if d else float('inf')
                
                # Calculate new distance after swap
                new_dist = 0
                d, _ = dijkstra_shortest_path(env, route[i], route[j-1], distance_cache)
                new_dist += d if d else float('inf')
                d, _ = dijkstra_shortest_path(env, route[i+1], route[j], distance_cache)
                new_dist += d if d else float('inf')
                
                # If improvement, do the swap (reverse segment between i+1 and j-1)
                if new_dist < current_dist:
                    route[i+1:j] = reversed(route[i+1:j])
                    improved = True
    
    return route


def optimize_vehicle_selection(env, orders, vehicle_states):
    """
    Minimize number of vehicles by aggressive packing
    
    Strategy:
    1. Sort vehicles by capacity (largest first)
    2. Sort orders by size (largest first) 
    3. Greedily pack orders into fewest vehicles
    
    Returns: Better vehicle assignments
    """
    # Get vehicle capacities
    vehicles_by_capacity = []
    for vehicle in env.get_all_vehicles():
        vehicles_by_capacity.append({
            'id': vehicle.id,
            'weight_cap': vehicle.weight_capacity,
            'volume_cap': vehicle.volume_capacity,
            'home_wh': vehicle.home_warehouse_id,
            'used_weight': 0,
            'used_volume': 0,
            'orders': []
        })
    
    # Sort by capacity (largest first)
    vehicles_by_capacity.sort(key=lambda v: v['weight_cap'] * v['volume_cap'], reverse=True)
    
    # Sort orders by size (largest first - pack big items first)
    order_sizes = []
    for order_id in orders:
        reqs = env.get_order_requirements(order_id)
        total_weight = sum(env.skus[sku].weight * qty for sku, qty in reqs.items())
        total_volume = sum(env.skus[sku].volume * qty for sku, qty in reqs.items())
        order_sizes.append({
            'id': order_id,
            'weight': total_weight,
            'volume': total_volume,
            'size': total_weight * total_volume
        })
    
    order_sizes.sort(key=lambda o: o['size'], reverse=True)
    
    # Greedy packing
    for order in order_sizes:
        placed = False
        # Try to fit in existing vehicles first
        for vehicle in vehicles_by_capacity:
            if (vehicle['used_weight'] + order['weight'] <= vehicle['weight_cap'] and
                vehicle['used_volume'] + order['volume'] <= vehicle['volume_cap']):
                vehicle['orders'].append(order['id'])
                vehicle['used_weight'] += order['weight']
                vehicle['used_volume'] += order['volume']
                placed = True
                break
        
        if not placed:
            # Need new vehicle - this means we can't reduce vehicle count further
            pass
    
    return vehicles_by_capacity


def calculate_route_distance_fast(steps, env):
    """Fast route distance calculation"""
    total = 0
    for i in range(len(steps) - 1):
        d = env.get_distance(steps[i]['node_id'], steps[i+1]['node_id'])
        if d:
            total += d
    return total


# Usage in main solver:
"""
# After TSP ordering of delivery nodes:
delivery_nodes = two_opt_route_optimization(delivery_nodes, env, distance_cache)

# Before building routes:
optimal_vehicle_assignment = optimize_vehicle_selection(env, all_order_ids, vehicle_states)
"""
