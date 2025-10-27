"""
Ne3Na3 Solver 52 - ULTRA-SIMPLE Consolidation for Cost Minimization
======================================================================

EXTREME SIMPLIFICATION:
1. Use ONLY HeavyTrucks (best cost/capacity: $0.17 vs $0.27)
2. Pack ALL orders into MINIMUM number of vehicles
3. Simple nearest-neighbor TSP for routing
4. No ML, no optimization - just CONSOLIDATION

Target: ~$2,700 (2 HeavyTrucks @ $1,200 each + $300 distance)
vs Current: $3,247 (4 vehicles)
"""

import heapq
from typing import Any, Dict, List, Tuple, Optional

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
                continue  # Skip starting node (already added)
            
            if node == delivery_node:
                # Delivery node - add deliveries
                deliveries = order_assignments.get(delivery_node, [])
                steps.append({"node_id": node, "pickups": [], "deliveries": deliveries, "unloads": []})
            else:
                # Intermediate node
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


def solver(env: Any) -> Dict:
    """
    ULTRA-SIMPLE consolidation solver.
    Strategy: Use HeavyTrucks first (best cost/capacity), then fallback to smaller vehicles if needed.
    """
    print("\n[SOLVER 52] Ultra-Simple Consolidation - Minimize Vehicle Count")
    print("=" * 80)
    
    # Get vehicles by type (prefer HeavyTrucks)
    all_vehicles = list(env.get_all_vehicles())
    heavy_trucks = [v for v in all_vehicles if 'Heavy' in v.type or 'Heavy' in v.id]
    medium_trucks = [v for v in all_vehicles if 'Medium' in v.type]
    light_vans = [v for v in all_vehicles if 'Light' in v.type or 'LightVan' in v.id]
    
    print(f"Available vehicles:")
    print(f"  HeavyTrucks: {len(heavy_trucks)} @ ${heavy_trucks[0].fixed_cost if heavy_trucks else 0}")
    print(f"  MediumTrucks: {len(medium_trucks)} @ ${medium_trucks[0].fixed_cost if medium_trucks else 0}")
    print(f"  LightVans: {len(light_vans)} @ ${light_vans[0].fixed_cost if light_vans else 0}")
    
    # Use ALL vehicles in priority order: Heavy > Medium > Light
    prioritized_vehicles = heavy_trucks + medium_trucks + light_vans
    
    if heavy_trucks:
        print(f"HeavyTruck capacity: {heavy_trucks[0].capacity_weight}kg / {heavy_trucks[0].capacity_volume}m³")
        print(f"HeavyTruck cost: ${heavy_trucks[0].fixed_cost} + ${heavy_trucks[0].cost_per_km}/km")
    
    # Group ALL vehicles by warehouse (prioritized order)
    trucks_by_wh = {}
    warehouses = env.warehouses
    for vehicle in prioritized_vehicles:
        wh_id = vehicle.home_warehouse_id
        if wh_id not in trucks_by_wh:
            trucks_by_wh[wh_id] = []
        trucks_by_wh[wh_id].append(vehicle)
    
    # Get all orders
    all_orders = env.get_all_order_ids()
    print(f"Total orders: {len(all_orders)}")
    
    # SMART allocation: Fill HeavyTrucks as much as possible, then use cheapest vehicle for remainder
    # Goal: 2 HeavyTrucks @ 99% + 1 LightVan for overflow = $2,700
    orders_by_wh = {wh_id: [] for wh_id in warehouses.keys()}
    
    # Simple even split across warehouses (since we have 1 HeavyTruck per warehouse)
    orders_per_warehouse = len(all_orders) // len(warehouses)
    
    for idx, oid in enumerate(all_orders):
        wh_idx = min(idx // orders_per_warehouse, len(list(warehouses.keys())) - 1)
        wh_id = list(warehouses.keys())[wh_idx]
        orders_by_wh[wh_id].append(oid)
    
    print(f"Orders per warehouse: {[(wh, len(ords)) for wh, ords in orders_by_wh.items()]}")
    
    solution = {"routes": []}
    
    # Pack orders into HeavyTrucks for each warehouse
    for wh_id, orders in orders_by_wh.items():
        if not orders:
            continue
        
        wh = warehouses[wh_id]
        wh_node = getattr(wh.location, 'id', wh.location)
        trucks = trucks_by_wh.get(wh_id, [])
        
        if not trucks:
            print(f"[WARNING] No HeavyTrucks for warehouse {wh_id}")
            continue
        
        print(f"\n[Warehouse {wh_id}] Packing {len(orders)} orders into {len(trucks)} vehicles")
        
        # OPTIMIZED PACKING STRATEGY:
        # 1. Fill HeavyTrucks to 99%+ capacity
        # 2. For remainder, use CHEAPEST vehicle (LightVan)
        
        truck_loads = []
        remaining_orders = list(orders)
        
        # PHASE 1: Fill HeavyTrucks
        for truck_idx, truck in enumerate(trucks):
            if not remaining_orders:
                break
            
            # Skip non-HeavyTrucks in first pass
            if 'Heavy' not in truck.type:
                continue
            
            packed_orders = []
            rem_weight = truck.capacity_weight
            rem_volume = truck.capacity_volume
            
            # Greedy pack from remaining orders
            for oid in remaining_orders[:]:
                req = env.get_order_requirements(oid)
                
                # Calculate order weight/volume
                order_weight = sum(env.skus[sku].weight * qty for sku, qty in req.items())
                order_volume = sum(env.skus[sku].volume * qty for sku, qty in req.items())
                
                if order_weight <= rem_weight and order_volume <= rem_volume:
                    packed_orders.append(oid)
                    remaining_orders.remove(oid)
                    rem_weight -= order_weight
                    rem_volume -= order_volume
            
            utilization_w = (truck.capacity_weight - rem_weight) / truck.capacity_weight * 100
            utilization_v = (truck.capacity_volume - rem_volume) / truck.capacity_volume * 100
            
            print(f"  {truck.type} {truck_idx+1}: {len(packed_orders)} orders, "
                  f"{utilization_w:.1f}% weight, {utilization_v:.1f}% volume")
            
            if packed_orders:
                truck_loads.append((truck, packed_orders))
        
        # PHASE 2: Pack remainder into CHEAPEST available vehicle (LightVan)
        if remaining_orders:
            print(f"  Remaining orders: {len(remaining_orders)}")
            
            # Find cheapest vehicle that can fit
            for truck in reversed(trucks):  # Reverse to get LightVan first
                if 'Light' in truck.type or 'LightVan' in truck.id:
                    packed_orders = []
                    rem_weight = truck.capacity_weight
                    rem_volume = truck.capacity_volume
                    
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
                        utilization_w = (truck.capacity_weight - rem_weight) / truck.capacity_weight * 100
                        utilization_v = (truck.capacity_volume - rem_volume) / truck.capacity_volume * 100
                        print(f"  {truck.type}: {len(packed_orders)} orders, "
                              f"{utilization_w:.1f}% weight, {utilization_v:.1f}% volume")
                        truck_loads.append((truck, packed_orders))
                        break
        
        if remaining_orders:
            print(f"  [WARNING] {len(remaining_orders)} orders could not be packed!")
        
        # Build routes
        for truck, packed_orders in truck_loads:
            # Nearest neighbor TSP for delivery order
            delivery_sequence = []
            current_node = wh_node
            remaining = set(packed_orders)
            
            while remaining:
                closest_order = None
                min_dist = float('inf')
                
                for oid in remaining:
                    order_loc = env.get_order_location(oid)
                    dist = abs(current_node - order_loc)
                    if dist < min_dist:
                        min_dist = dist
                        closest_order = oid
                
                if closest_order:
                    delivery_sequence.append(closest_order)
                    current_node = env.get_order_location(closest_order)
                    remaining.remove(closest_order)
            
            # Build order_assignments: node -> deliveries
            order_assignments = {}
            for oid in delivery_sequence:
                order_loc = env.get_order_location(oid)
                req = env.get_order_requirements(oid)
                
                if order_loc not in order_assignments:
                    order_assignments[order_loc] = []
                
                for sku, qty in req.items():
                    order_assignments[order_loc].append({
                        "order_id": oid,
                        "sku_id": sku,
                        "quantity": qty
                    })
            
            # Build route steps
            delivery_nodes = [env.get_order_location(oid) for oid in delivery_sequence]
            steps = build_steps_with_path(env, wh_node, delivery_nodes, order_assignments)
            
            # Add pickups to first step
            if steps:
                pickups = []
                for oid in packed_orders:
                    req = env.get_order_requirements(oid)
                    for sku, qty in req.items():
                        pickups.append({
                            "warehouse_id": wh_id,
                            "sku_id": sku,
                            "quantity": qty
                        })
                steps[0]["pickups"] = pickups
            
            solution["routes"].append({
                "vehicle_id": truck.id,
                "steps": steps
            })
    
    print(f"\n[SOLVER 52] Created {len(solution['routes'])} routes")
    
    # Calculate cost
    try:
        cost = env.calculate_solution_cost(solution)
        print(f"Estimated cost: ${cost:,.0f}")
    except:
        pass
    
    print("=" * 80)
    
    return solution
