#!/usr/bin/env python3
"""
Ne3Na3 Final Solver - Balanced Performance with Per-Run Caching

Strategy:
- Per-run caching (cleared each solver call - ALLOWED by rules)
- Greedy warehouse assignment with connectivity check  
- Capacity-aware vehicle packing
- Simple route building without expensive optimizations
"""
from robin_logistics import LogisticsEnvironment
from typing import Dict, Tuple, Optional
import heapq


def dijkstra_with_path(env, start, end):
    """Dijkstra returning (distance, path). NO cross-run caching."""
    if start == end:
        return 0.0, [start]
    
    road = env.get_road_network_data()
    adj = road.get("adjacency_list", {})
    
    if start not in adj or end not in adj:
        return None, None
    
    dist = {start: 0.0}
    prev = {start: None}
    visited = set()
    pq = [(0.0, start)]
    
    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        
        if node == end:
            # Reconstruct path
            path = []
            curr = end
            while curr is not None:
                path.append(curr)
                curr = prev.get(curr)
            path.reverse()
            return d, path
        
        for neighbor in adj.get(node, []):
            if neighbor in visited:
                continue
            edge_dist = env.get_distance(node, neighbor)
            if edge_dist is None:
                continue
            new_dist = d + edge_dist
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))
    
    return None, None


def solver(env) -> Dict:
    """Main solver with per-run caching."""
    
    # PER-RUN CACHE (local to this function call)
    distance_cache = {}
    path_cache = {}
    
    def get_distance(start, end):
        """Get distance with per-run caching."""
        if start == end:
            return 0.0
        key = (start, end)
        if key not in distance_cache:
            d, _ = dijkstra_with_path(env, start, end)
            distance_cache[key] = d
        return distance_cache[key]
    
    def get_path(start, end):
        """Get path with per-run caching."""
        if start == end:
            return [start]
        key = (start, end)
        if key not in path_cache:
            _, p = dijkstra_with_path(env, start, end)
            path_cache[key] = p
        return path_cache[key]
    
    # Extract data
    orders = []
    for order_id in env.get_all_order_ids():
        order = env.orders[order_id]
        orders.append({
            'id': order_id,
            'node': order.destination.id,
            'items': dict(order.requested_items),
            'weight': sum(env.skus[sku].weight * qty for sku, qty in order.requested_items.items()),
            'volume': sum(env.skus[sku].volume * qty for sku, qty in order.requested_items.items())
        })
    
    warehouses = []
    for wh_id, wh in env.warehouses.items():
        warehouses.append({
            'id': wh_id,
            'node': wh.location.id,
            'inventory': dict(wh.inventory)
        })
    
    vehicles = []
    for wh_id, wh in env.warehouses.items():
        for vehicle in wh.vehicles:
            vehicles.append({
                'id': vehicle.id,
                'wh_id': wh_id,
                'wh_node': wh.location.id,
                'cap_weight': vehicle.capacity_weight,
                'cap_volume': vehicle.capacity_volume
            })
    
    # Assign orders to warehouses (greedy nearest with connectivity)
    assigned = {}
    for order in orders:
        best_wh = None
        best_dist = float('inf')
        
        for wh in warehouses:
            can_fulfill = all(wh['inventory'].get(sku, 0) >= qty for sku, qty in order['items'].items())
            if can_fulfill:
                dist = get_distance(wh['node'], order['node'])
                if dist is not None and dist < best_dist:
                    best_dist = dist
                    best_wh = wh
        
        if best_wh:
            assigned[order['id']] = best_wh['id']
            for sku, qty in order['items'].items():
                best_wh['inventory'][sku] -= qty
    
    # Pack orders into vehicles
    routes = []
    for vehicle in vehicles:
        vehicle_orders = [o for o in orders if assigned.get(o['id']) == vehicle['wh_id']]
        if not vehicle_orders:
            continue
        
        # Pack what fits
        packed = []
        curr_weight = 0.0
        curr_volume = 0.0
        
        for order in vehicle_orders:
            if (curr_weight + order['weight'] <= vehicle['cap_weight'] and
                curr_volume + order['volume'] <= vehicle['cap_volume']):
                # Verify connectivity
                dist = get_distance(vehicle['wh_node'], order['node'])
                if dist is not None:
                    packed.append(order)
                    curr_weight += order['weight']
                    curr_volume += order['volume']
        
        if not packed:
            continue
        
        # Build route with INTERMEDIATE NODES from paths
        steps = []
        
        # Start at warehouse
        steps.append({
            'node_id': vehicle['wh_node'],
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
        
        # Pickup at warehouse
        pickups = []
        for order in packed:
            for sku, qty in order['items'].items():
                pickups.append({
                    'warehouse_id': vehicle['wh_id'],
                    'sku_id': sku,
                    'quantity': qty
                })
        
        steps.append({
            'node_id': vehicle['wh_node'],
            'pickups': pickups,
            'deliveries': [],
            'unloads': []
        })
        
        # Deliver to each order (with intermediate nodes)
        current_node = vehicle['wh_node']
        for order in packed:
            path = get_path(current_node, order['node'])
            if path is None or len(path) < 2:
                continue
            
            # Add intermediate nodes
            for intermediate in path[1:-1]:
                steps.append({
                    'node_id': intermediate,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
            
            # Add delivery at destination
            deliveries = [
                {'order_id': order['id'], 'sku_id': sku, 'quantity': qty}
                for sku, qty in order['items'].items()
            ]
            steps.append({
                'node_id': order['node'],
                'pickups': [],
                'deliveries': deliveries,
                'unloads': []
            })
            current_node = order['node']
        
        # Return to warehouse (with intermediate nodes)
        path_home = get_path(current_node, vehicle['wh_node'])
        if path_home and len(path_home) > 1:
            for intermediate in path_home[1:]:
                steps.append({
                    'node_id': intermediate,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
        
        routes.append({'vehicle_id': vehicle['id'], 'steps': steps})
    
    return {"routes": routes}


if __name__ == '__main__':
    env = LogisticsEnvironment()
    result = solver(env)
    print(f"Routes: {len(result['routes'])}")
