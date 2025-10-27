#!/usr/bin/env python3
"""
Ne3Na3 Team - Multi-Order NAR Solver
Submission 1 for Beltone AI Hackathon

Strategy: Multi-order consolidation with capacity-aware packing
- Group multiple orders per vehicle to maximize fulfillment
- Cluster orders by warehouse proximity
- Build efficient multi-stop routes
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple, Set
from collections import deque, defaultdict
import math


# Global distance cache for this run
_distance_cache = {}


def get_path_distance(env: LogisticsEnvironment, start_node: int, end_node: int) -> Optional[float]:
    """Get distance between two nodes using BFS pathfinding with caching."""
    if start_node == end_node:
        return 0.0
    
    cache_key = (start_node, end_node)
    if cache_key in _distance_cache:
        return _distance_cache[cache_key]
    
    direct_dist = env.get_distance(start_node, end_node)
    if direct_dist is not None:
        _distance_cache[cache_key] = direct_dist
        return direct_dist
    
    path = find_shortest_path(env, start_node, end_node)
    if not path or len(path) < 2:
        _distance_cache[cache_key] = None
        return None
    
    total_dist = 0.0
    for i in range(len(path) - 1):
        segment_dist = env.get_distance(path[i], path[i + 1])
        if segment_dist is None:
            _distance_cache[cache_key] = None
            return None
        total_dist += segment_dist
    
    _distance_cache[cache_key] = total_dist
    return total_dist


def solver(env: LogisticsEnvironment) -> Dict:
    solution = {"routes": []}
    
    order_ids = env.get_all_order_ids()
    vehicles = env.get_all_vehicles()
    warehouses = env.warehouses
    orders = env.orders
    skus = env.skus
    
    fulfilled_orders = set()
    
    warehouse_order_groups = defaultdict(list)
    
    for order_id in order_ids:
        order = orders[order_id]
        # find nearest warehouse
        best_wh = None
        min_dist = float('inf')
        
        for wh_id, wh in warehouses.items():
            can_fulfill = True
            for sku_id, qty in order.requested_items.items():
                if wh.inventory.get(sku_id, 0) < qty:
                    can_fulfill = False
                    break
            
            if can_fulfill:
                dist = get_path_distance(env, wh.location.id, order.destination.id)
                if dist is not None and dist < min_dist:
                    min_dist = dist
                    best_wh = wh_id
        
        if best_wh:
            warehouse_order_groups[best_wh].append((order_id, min_dist))
    
    for wh_id in warehouse_order_groups:
        warehouse_order_groups[wh_id].sort(key=lambda x: x[1])
    
    # assign vehicles to order batches
    for vehicle in vehicles:
        if len(fulfilled_orders) >= 45:  # Target reached
            break
        
        wh_id = vehicle.home_warehouse_id
        
        available_orders = [
            oid for oid, _ in warehouse_order_groups.get(wh_id, [])
            if oid not in fulfilled_orders
        ]
        
        if not available_orders:
            continue
        
        # pack as many orders as possible into this vehicle
        order_batch = pack_orders_into_vehicle(
            env, vehicle, wh_id, available_orders, fulfilled_orders
        )
        
        if not order_batch:
            continue
        
        # build multi-stop route for this vehicle
        route = build_multi_order_route(env, vehicle, wh_id, order_batch)
        
        if route and route['steps']:
            solution['routes'].append(route)
            fulfilled_orders.update(order_batch.keys())
    
    return solution


def pack_orders_into_vehicle(
    env: LogisticsEnvironment,
    vehicle: object,
    warehouse_id: str,
    available_order_ids: List[str],
    fulfilled_orders: Set[str]
) -> Dict[str, Dict[str, int]]:
    """
    Pack multiple orders into a vehicle respecting capacity constraints.
    Returns: {order_id: {sku_id: quantity}}
    """
    warehouse = env.warehouses[warehouse_id]
    orders = env.orders
    skus = env.skus
    
    packed_orders = {}
    current_weight = 0.0
    current_volume = 0.0
    
    for order_id in available_order_ids:
        if order_id in fulfilled_orders:
            continue
        
        order = orders[order_id]
        
        # Calculate order requirements
        order_weight = sum(
            skus[sku_id].weight * qty
            for sku_id, qty in order.requested_items.items()
        )
        order_volume = sum(
            skus[sku_id].volume * qty
            for sku_id, qty in order.requested_items.items()
        )
        
        if (current_weight + order_weight <= vehicle.capacity_weight and
            current_volume + order_volume <= vehicle.capacity_volume):
            
            can_fulfill = True
            for sku_id, qty in order.requested_items.items():
                available = warehouse.inventory.get(sku_id, 0)
                already_allocated = sum(
                    packed_orders[oid].get(sku_id, 0)
                    for oid in packed_orders
                )
                if available < qty + already_allocated:
                    can_fulfill = False
                    break
            
            if can_fulfill:
                packed_orders[order_id] = dict(order.requested_items)
                current_weight += order_weight
                current_volume += order_volume
    
    return packed_orders


def build_multi_order_route(
    env: LogisticsEnvironment,
    vehicle: object,
    warehouse_id: str,
    order_batch: Dict[str, Dict[str, int]]
) -> Dict:
    """
    Build route for vehicle serving multiple orders.
    Route: home → warehouse (pickup) → delivery1 → delivery2 → ... → home
    """
    if not order_batch:
        return None
    
    warehouse = env.warehouses[warehouse_id]
    orders = env.orders
    home_node = warehouse.location.id
    
    # Collect all delivery locations
    delivery_locations = [
        (order_id, orders[order_id].destination.id)
        for order_id in order_batch
    ]
    
    # Optimize delivery order using nearest neighbor
    delivery_sequence = optimize_delivery_sequence(
        env, home_node, delivery_locations
    )
    
    steps = []
    current_node = home_node
    
    # Step 1: Start at home
    steps.append({
        'node_id': home_node,
        'pickups': [],
        'deliveries': [],
        'unloads': []
    })
    
    # Step 2: Pickup all items at warehouse
    pickup_list = []
    for order_id, sku_dict in order_batch.items():
        for sku_id, qty in sku_dict.items():
            pickup_list.append({
                'warehouse_id': warehouse_id,
                'sku_id': sku_id,
                'quantity': qty
            })
    
    if home_node != warehouse.location.id:
        path = find_shortest_path(env, home_node, warehouse.location.id)
        if path:
            for node in path[1:-1]:
                steps.append({
                    'node_id': node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
        current_node = warehouse.location.id
    
    steps.append({
        'node_id': warehouse.location.id,
        'pickups': pickup_list,
        'deliveries': [],
        'unloads': []
    })
    
    # Step 3: Deliver to each order location in optimized sequence
    for order_id, delivery_node in delivery_sequence:
        path = find_shortest_path(env, current_node, delivery_node)
        if not path:
            continue
        
        # Add intermediate nodes
        for node in path[1:-1]:
            steps.append({
                'node_id': node,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })
        
        # Add delivery step
        delivery_list = [
            {'order_id': order_id, 'sku_id': sku_id, 'quantity': qty}
            for sku_id, qty in order_batch[order_id].items()
        ]
        
        steps.append({
            'node_id': delivery_node,
            'pickups': [],
            'deliveries': delivery_list,
            'unloads': []
        })
        
        current_node = delivery_node
    
    # Step 4: Return home
    path_home = find_shortest_path(env, current_node, home_node)
    if path_home:
        for node in path_home[1:]:
            steps.append({
                'node_id': node,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })
    
    # Clean up duplicate consecutive nodes
    steps = remove_duplicate_consecutive_nodes(steps)
    
    return {
        'vehicle_id': vehicle.id,
        'steps': steps
    }


def optimize_delivery_sequence(
    env: LogisticsEnvironment,
    start_node: int,
    delivery_locations: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    """
    Optimize delivery sequence using nearest neighbor heuristic.
    Returns list of (order_id, node_id) in optimized order.
    """
    if not delivery_locations:
        return []
    
    remaining = list(delivery_locations)
    sequence = []
    current = start_node
    
    while remaining:
        # Find nearest unvisited delivery
        best_idx = 0
        best_dist = float('inf')
        
        for idx, (order_id, node_id) in enumerate(remaining):
            dist = get_path_distance(env, current, node_id)
            if dist is not None and dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        # Add to sequence
        next_delivery = remaining.pop(best_idx)
        sequence.append(next_delivery)
        current = next_delivery[1]
    
    return sequence


def find_shortest_path(
    env: LogisticsEnvironment,
    start_node: int,
    end_node: int
) -> Optional[List[int]]:
    """Find shortest path using BFS on road network."""
    if start_node == end_node:
        return [start_node]
    
    road_network = env.get_road_network_data()
    adjacency_list = road_network.get('adjacency_list', {})
    
    if start_node not in adjacency_list:
        return None
    
    queue = deque([(start_node, [start_node])])
    visited = {start_node}
    
    while queue:
        current_node, path = queue.popleft()
        neighbors = adjacency_list.get(current_node, [])
        
        for neighbor_id in neighbors:
            if neighbor_id == end_node:
                return path + [neighbor_id]
            
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append((neighbor_id, path + [neighbor_id]))
    
    return None


def remove_duplicate_consecutive_nodes(steps: List[Dict]) -> List[Dict]:
    """Remove consecutive steps with same node_id, merging operations."""
    if not steps:
        return steps
    
    cleaned = [steps[0]]
    
    for step in steps[1:]:
        if step['node_id'] == cleaned[-1]['node_id']:
            cleaned[-1]['pickups'].extend(step['pickups'])
            cleaned[-1]['deliveries'].extend(step['deliveries'])
            cleaned[-1]['unloads'].extend(step['unloads'])
        else:
            cleaned.append(step)
    
    return cleaned


# if __name__ == '__main__':
#     env = LogisticsEnvironment()
#     solution = solver(env)
#     
#     # Validate and display results
#     validation_result = env.validate_solution_complete(solution)
#     print(f"Validation Result: {validation_result}")
#     
#     # Try executing the solution
#     try:
#         success, message = env.execute_solution(solution)
#         print(f"\nExecution Success: {success}")
#         print(f"Message: {message}")
#         
#         if success:
#             cost = env.calculate_solution_cost(solution)
#             stats = env.get_solution_statistics(solution)
#             fulfillment = env.get_solution_fulfillment_summary(solution)
#             
#             print(f"\n=== Solution Statistics ===")
#             print(f"Total Cost: ${cost:,.2f}")
#             print(f"Routes: {len(solution['routes'])}")
#             print(f"\nFulfillment Summary:")
#             print(f"  Total Orders: {fulfillment['total_orders']}")
#             print(f"  Orders Served: {fulfillment['orders_served']}")
#             print(f"  Fully Fulfilled: {fulfillment['fully_fulfilled_orders']}")
#             print(f"  Fulfillment Rate: {fulfillment['average_fulfillment_rate']:.1f}%")
#             print(f"  Vehicle Utilization: {fulfillment['vehicle_utilization']*100:.1f}%")
#             print(f"\nDistance: {stats['total_distance']:.2f} km")
#             print(f"Cost Breakdown:")
#             print(f"  Fixed: ${stats['fixed_cost_total']:,.2f}")
#             print(f"  Variable: ${stats['variable_cost_total']:,.2f}")
#     except Exception as e:
#         print(f"Error executing solution: {e}")
#         import traceback
#         traceback.print_exc()
