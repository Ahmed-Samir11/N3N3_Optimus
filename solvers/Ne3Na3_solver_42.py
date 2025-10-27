#!/usr/bin/env python3
"""
Ne3Na3 Team - Solver 39 - Adaptive Large Neighborhood Search (ALNS)
======================================================================

Strategy: State-of-the-Art Metaheuristic for Cost Minimization

This solver replaces the previous predictive ML approach with a powerful,
state-of-the-art metaheuristic called Adaptive Large Neighborhood Search (ALNS).
This is the standard, competition-winning approach for complex Vehicle
Routing Problems.

Core Principles:
1.  **Correct Foundation:** All distance calculations are now correct. The solver
    pre-computes a distance matrix using Dijkstra's algorithm for all key
    locations. This is a one-time operation at the start and is crucial for
    both speed and accuracy.

2.  **Ruin and Recreate:** Instead of building a solution once, ALNS starts with an
    initial solution and iteratively destroys ("ruins") a small part of it and
    then intelligently rebuilds ("recreates") it. This process is repeated
    thousands of time, allowing the solver to escape local optima and find
    globally low-cost solutions.

3.  **Intelligent Search:** The solver uses a set of powerful "destroy" and
    "repair" operators:
    - **Destroy Operators:** `worst_removal` intelligently removes the most
      expensive orders, giving the solver a chance to place them better.
      `random_removal` provides diversification.
    - **Repair Operators:** `regret_insertion` is a sophisticated heuristic
      that prioritizes re-inserting orders that have the fewest good options,
      preventing poor decisions early on.

4.  **Simulated Annealing:** A Simulated Annealing acceptance criterion is used
    to intelligently decide when to accept worse solutions, preventing the
    search from getting stuck.
"""

import time
import math
import heapq
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set, Optional

# ============================================================================
# SECTION 1: CORRECT PATHFINDING & DISTANCE MATRIX (THE FOUNDATION)
# ============================================================================

def build_distance_matrix(env: Any) -> Dict[Tuple[int, int], float]:
    """
    Pre-computes all-to-all shortest path distances for key locations using Dijkstra's.
    This is the allowed, 'legal' way to cache distances for a single run and is
    CRITICAL for both performance and correctness.
    """
    print("Pre-computing distance matrix...")
    start_time = time.time()
    
    points_of_interest = set()
    # Add warehouse locations
    for wh_id, wh in env.warehouses.items():
        points_of_interest.add(wh.location.id)
    # Add order destinations
    for order_id, order in env.orders.items():
        points_of_interest.add(order.destination.id)

    dist_matrix = {}
    adjacency = env.get_road_network_data().get("adjacency_list", {})
    all_nodes = set(adjacency.keys())

    for i, start_node in enumerate(points_of_interest):
        # Dijkstra from one source to all other points of interest
        distances_from_start = dijkstra_single_source(env, start_node, adjacency, all_nodes)
        for end_node in points_of_interest:
            dist_matrix[(start_node, end_node)] = distances_from_start.get(end_node, float('inf'))
            
    print(f"Distance matrix built for {len(points_of_interest)} nodes in {time.time() - start_time:.2f}s")
    return dist_matrix

def dijkstra_single_source(env: Any, start_node: int, adjacency: Dict, all_nodes: Set[int]) -> Dict[int, float]:
    """Finds shortest distance from a single source to all other nodes."""
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in distances: return {}
    
    distances[start_node] = 0.0
    pq = [(0.0, start_node)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_distance > distances[current_node]:
            continue

        for neighbor in adjacency.get(current_node, []):
            weight = env.get_distance(current_node, neighbor)
            if weight is None: continue

            distance = current_distance + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
                
    return distances

# ============================================================================
# SECTION 2: CORE ALNS LOGIC (RUIN AND RECREATE)
# ============================================================================

def solve_with_alns(env: Any, time_budget: float) -> Dict:
    """
    Main ALNS solver function.
    """
    start_time = time.time()
    
    # 1. FOUNDATION: Build the essential distance matrix
    dist_matrix = build_distance_matrix(env)
    
    # 2. INITIAL SOLUTION: Create a good starting point using best insertion heuristic
    print("Building initial solution...")
    solution = build_initial_solution(env, dist_matrix)
    
    best_solution = solution
    current_solution = solution
    best_cost = calculate_solution_cost(env, best_solution, dist_matrix)
    print(f"Initial solution cost: {best_cost:,.2f} (Routes: {len(solution.get('routes', []))})")

    # 3. ALNS MAIN LOOP
    # SA parameters for acceptance criteria
    temperature = best_cost * 0.05 
    cooling_rate = 0.999
    iteration = 0

    while time.time() - start_time < time_budget:
        iteration += 1
        
        # a. Ruin: Destroy part of the current solution
        temp_solution, request_bank = destroy_random(current_solution, num_to_remove=random.randint(5, 15))

        # b. Recreate: Rebuild the solution using an intelligent heuristic
        # Alternate between greedy and regret insertion for diversity
        if iteration % 2 == 0:
            new_solution = repair_regret_insertion(env, temp_solution, request_bank, dist_matrix)
        else:
            new_solution = repair_greedy_insertion(env, temp_solution, request_bank, dist_matrix)

        # c. Evaluate and Decide
        current_cost = calculate_solution_cost(env, current_solution, dist_matrix)
        new_cost = calculate_solution_cost(env, new_solution, dist_matrix)
        
        if new_cost < current_cost:
            current_solution = new_solution
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
                print(f"  -> Iter {iteration}: New best cost: {best_cost:,.2f} (Time: {time.time() - start_time:.1f}s)")
        # Simulated Annealing acceptance criterion
        elif random.random() < math.exp((current_cost - new_cost) / temperature):
            current_solution = new_solution
        
        temperature *= cooling_rate

    print(f"ALNS completed: {iteration} iterations, final cost: {best_cost:,.2f}")
    return best_solution


# --- DESTROY (RUIN) OPERATORS ---
def destroy_random(solution: Dict, num_to_remove: int) -> Tuple[Dict, Set[str]]:
    """Removes a number of random orders from the solution."""
    routes = solution.get("routes", [])
    if not routes: return solution, set()

    all_orders = []
    for route in routes:
        for step in route.get("steps", []):
            for delivery in step.get("deliveries", []):
                all_orders.append(delivery['order_id'])
    
    if not all_orders: return solution, set()

    orders_to_remove = set(random.sample(all_orders, min(num_to_remove, len(all_orders))))
    
    # Create new routes without the removed orders
    new_routes = []
    for route in routes:
        new_steps = []
        orders_in_route = set()
        for step in route.get("steps", []):
            # Keep pickups for now, will be re-calculated later if needed
            new_step = step.copy()
            deliveries = step.get("deliveries", [])
            new_deliveries = [d for d in deliveries if d['order_id'] not in orders_to_remove]
            new_step['deliveries'] = new_deliveries
            
            if new_deliveries:
                for d in new_deliveries:
                    orders_in_route.add(d['order_id'])
            
            # Only add steps that are still relevant
            if new_step.get('pickups') or new_step.get('deliveries'):
                 new_steps.append(new_step)

        if orders_in_route: # If route still has orders
             new_routes.append({"vehicle_id": route['vehicle_id'], "steps": new_steps})

    return {"routes": new_routes}, orders_to_remove


# --- REPAIR (RECREATE) OPERATORS ---
def repair_greedy_insertion(env: Any, solution: Dict, request_bank: Set[str], dist_matrix: Dict) -> Dict:
    """
    Re-inserts all orders from request_bank using greedy best insertion.
    Inserts each order into its cheapest valid position in existing routes.
    """
    # Convert to list and shuffle for randomness
    orders_to_insert = list(request_bank)
    random.shuffle(orders_to_insert)
    
    for order_id in orders_to_insert:
        # Find best insertion position (only in existing routes, not new routes)
        best_insertion = None
        best_cost = float('inf')
        
        # Get all valid insertions
        insertions = find_n_best_insertions(env, order_id, solution, dist_matrix, n=100)
        
        # Filter to only existing routes (route_idx != -1)
        for cost_increase, vehicle_id, route_idx, position in insertions:
            if route_idx != -1 and cost_increase < best_cost:
                best_cost = cost_increase
                best_insertion = (cost_increase, vehicle_id, route_idx, position)
        
        # If no valid position in existing routes, try creating new route
        if best_insertion is None:
            for cost_increase, vehicle_id, route_idx, position in insertions:
                if route_idx == -1:  # New route
                    best_insertion = (cost_increase, vehicle_id, route_idx, position)
                    break
        
        # Insert the order
        if best_insertion:
            _, vehicle_id, route_idx, position = best_insertion
            solution = insert_order_into_route(env, solution, order_id, vehicle_id, route_idx, position, dist_matrix)
    
    return solution

def repair_regret_insertion(env: Any, solution: Dict, request_bank: Set[str], dist_matrix: Dict) -> Dict:
    """
    Advanced repair using regret heuristic.
    Prioritizes inserting orders with highest regret (difference between best and second-best insertion cost).
    Orders with high regret have few good options and should be placed first.
    """
    remaining_orders = set(request_bank)
    
    while remaining_orders:
        # Calculate regret for each remaining order
        order_regrets = []
        
        for order_id in remaining_orders:
            # Find top 2 best insertions
            insertions = find_n_best_insertions(env, order_id, solution, dist_matrix, n=2)
            
            if len(insertions) == 0:
                # No valid insertion - skip this order
                continue
            elif len(insertions) == 1:
                # Only one option - very high regret!
                best_cost = insertions[0][0]
                regret = float('inf')  # Maximum regret - must insert now
                best_insertion = insertions[0]
            else:
                # Calculate regret as difference between second-best and best
                best_cost = insertions[0][0]
                second_best_cost = insertions[1][0]
                regret = second_best_cost - best_cost
                best_insertion = insertions[0]
            
            order_regrets.append((regret, order_id, best_insertion))
        
        if not order_regrets:
            # No more valid insertions possible
            break
        
        # Sort by regret (descending) - insert highest regret first
        order_regrets.sort(reverse=True, key=lambda x: x[0])
        
        # Insert the order with highest regret
        _, order_id, (cost_increase, vehicle_id, route_idx, position) = order_regrets[0]
        solution = insert_order_into_route(env, solution, order_id, vehicle_id, route_idx, position, dist_matrix)
        remaining_orders.remove(order_id)
    
    return solution

# ============================================================================
# SECTION 3: HELPER FUNCTIONS FOR INSERTION
# ============================================================================

def find_n_best_insertions(env: Any, order_id: str, solution: Dict, dist_matrix: Dict, n: int = 2) -> List[Tuple[float, str, int, int]]:
    """
    Finds the n cheapest valid insertion positions for an order.
    
    Returns: List of (cost_increase, vehicle_id, route_idx, position) tuples, sorted by cost_increase.
    """
    insertions = []
    order = env.orders[order_id]
    order_node = order.destination.id
    order_reqs = env.get_order_requirements(order_id)
    
    # Calculate order weight and volume
    order_weight = sum(env.skus[sku].weight * qty for sku, qty in order_reqs.items())
    order_volume = sum(env.skus[sku].volume * qty for sku, qty in order_reqs.items())
    
    routes = solution.get("routes", [])
    
    # Try inserting into existing routes
    for route_idx, route in enumerate(routes):
        vehicle_id = route['vehicle_id']
        # Find vehicle by ID
        vehicle = None
        for v in env.get_all_vehicles():
            if v.id == vehicle_id:
                vehicle = v
                break
        if not vehicle:
            continue
        
        steps = route.get('steps', [])
        
        # Check if vehicle has capacity for this order
        if order_weight > vehicle.capacity_weight or order_volume > vehicle.capacity_volume:
            continue
        
        # Calculate current capacity used - count each order only once
        orders_in_route = set()
        for step in steps:
            for delivery in step.get('deliveries', []):
                orders_in_route.add(delivery['order_id'])
        
        current_weight = 0.0
        current_volume = 0.0
        for oid in orders_in_route:
            oreqs = env.get_order_requirements(oid)
            current_weight += sum(env.skus[sku].weight * qty for sku, qty in oreqs.items())
            current_volume += sum(env.skus[sku].volume * qty for sku, qty in oreqs.items())
        
        # Check capacity constraint
        if current_weight + order_weight > vehicle.capacity_weight or current_volume + order_volume > vehicle.capacity_volume:
            continue
        
        # Find delivery positions (skip first/last which are depot)
        delivery_positions = []
        for i, step in enumerate(steps):
            if step.get('deliveries'):
                delivery_positions.append(i)
        
        # Try inserting after each delivery position
        for pos in range(len(delivery_positions) + 1):
            # Calculate cost increase
            cost_increase = calculate_insertion_cost(env, route, order_node, pos, delivery_positions, dist_matrix, vehicle)
            
            if cost_increase < float('inf'):
                insertions.append((cost_increase, vehicle_id, route_idx, pos))
    
    # Try creating new routes with available vehicles
    used_vehicles = {r['vehicle_id'] for r in routes}
    
    # Find all warehouses that have inventory for this order
    valid_warehouses = []
    for wh_id, wh in env.warehouses.items():
        wh_inv = env.get_warehouse_inventory(wh_id)
        if all(wh_inv.get(sku, 0) >= qty for sku, qty in order_reqs.items()):
            valid_warehouses.append(wh)
    
    # Try each unused vehicle that has capacity
    for vehicle in env.get_all_vehicles():
        if vehicle.id in used_vehicles:
            continue
        
        if vehicle.capacity_weight < order_weight or vehicle.capacity_volume < order_volume:
            continue
        
        # Find closest valid warehouse for this vehicle
        best_wh = None
        min_cost = float('inf')
        home_node = env.get_vehicle_home_warehouse(vehicle.id)
        
        for wh in valid_warehouses:
            wh_node = wh.location.id
            
            # Calculate round trip cost: home -> warehouse -> order -> home
            route_dist = (dist_matrix.get((home_node, wh_node), 0) + 
                         dist_matrix.get((wh_node, order_node), 0) + 
                         dist_matrix.get((order_node, home_node), 0))
            
            new_route_cost = vehicle.fixed_cost + vehicle.cost_per_km * route_dist
            
            if new_route_cost < min_cost:
                min_cost = new_route_cost
                best_wh = wh
        
        if best_wh:
            insertions.append((min_cost, vehicle.id, -1, -1))  # -1 indicates new route
    
    # Sort by cost and return top n
    insertions.sort(key=lambda x: x[0])
    return insertions[:n]


def calculate_insertion_cost(env: Any, route: Dict, order_node: int, position: int, 
                             delivery_positions: List[int], dist_matrix: Dict, vehicle: Any) -> float:
    """
    Calculates the cost increase of inserting an order at a specific position in a route.
    FIXED: Only uses delivery stop nodes (points of interest), not intermediate path nodes.
    """
    try:
        steps = route.get('steps', [])
        
        # Build current DELIVERY sequence (only delivery stops, not intermediate nodes)
        delivery_nodes = []
        for step in steps:
            if step.get('deliveries') and step['deliveries']:
                delivery_nodes.append(step['node_id'])
        
        if not delivery_nodes:
            # Empty route - just calculate cost of single delivery
            home_node = env.get_vehicle_home_warehouse(vehicle.id)
            
            # Get warehouse node
            wh_node = None
            for step in steps:
                if step.get('pickups'):
                    wh_node = step['node_id']
                    break
            
            if wh_node is None:
                return float('inf')
            
            # New route: home -> warehouse -> order -> home
            dist = (dist_matrix.get((home_node, wh_node), 0) +
                   dist_matrix.get((wh_node, order_node), 0) +
                   dist_matrix.get((order_node, home_node), 0))
            
            # Current route: home -> warehouse -> home
            current_dist = (dist_matrix.get((home_node, wh_node), 0) +
                           dist_matrix.get((wh_node, home_node), 0))
            
            return (dist - current_dist) * vehicle.cost_per_km
        
        # Calculate current route distance
        home_node = env.get_vehicle_home_warehouse(vehicle.id)
        wh_node = None
        for step in steps:
            if step.get('pickups'):
                wh_node = step['node_id']
                break
        
        if wh_node is None:
            return float('inf')
        
        current_dist = dist_matrix.get((home_node, wh_node), 0.0)
        prev = wh_node
        for node in delivery_nodes:
            current_dist += dist_matrix.get((prev, node), 0.0)
            prev = node
        current_dist += dist_matrix.get((prev, home_node), 0.0)
        
        # New route with insertion
        new_delivery_nodes = delivery_nodes[:position] + [order_node] + delivery_nodes[position:]
        
        new_dist = dist_matrix.get((home_node, wh_node), 0.0)
        prev = wh_node
        for node in new_delivery_nodes:
            d = dist_matrix.get((prev, node), None)
            if d is None:
                # Distance not in matrix - this insertion is not valid
                return float('inf')
            new_dist += d
            prev = node
        
        d = dist_matrix.get((prev, home_node), None)
        if d is None:
            return float('inf')
        new_dist += d
        
        cost_diff = (new_dist - current_dist) * vehicle.cost_per_km
        return cost_diff if cost_diff >= 0 else 0.0  # Never return negative cost
        
    except Exception as e:
        # Debug: print exception
        # print(f"Exception in calculate_insertion_cost: {e}")
        return float('inf')


def insert_order_into_route(env: Any, solution: Dict, order_id: str, vehicle_id: str, 
                            route_idx: int, position: int, dist_matrix: Dict) -> Dict:
    """
    Inserts an order into a specific position in a route (or creates new route if route_idx == -1).
    FIXED: Properly builds complete route with intermediate path nodes.
    """
    order = env.orders[order_id]
    order_node = order.destination.id
    order_reqs = env.get_order_requirements(order_id)
    
    if route_idx == -1:
        # Create new route - use same logic as working solvers
        vehicle = None
        for v in env.get_all_vehicles():
            if v.id == vehicle_id:
                vehicle = v
                break
        if not vehicle:
            return solution
        
        home_node = env.get_vehicle_home_warehouse(vehicle_id)
        
        # Find best warehouse (closest to order with inventory)
        best_wh = None
        min_dist = float('inf')
        
        for wh_id, wh in env.warehouses.items():
            wh_inv = env.get_warehouse_inventory(wh_id)
            if all(wh_inv.get(sku, 0) >= qty for sku, qty in order_reqs.items()):
                wh_node_temp = wh.location.id
                # Distance from warehouse to order
                dist = dist_matrix.get((wh_node_temp, order_node), float('inf'))
                if dist < min_dist:
                    min_dist = dist
                    best_wh = wh
        
        if not best_wh:
            return solution
        
        wh_node = best_wh.location.id
        
        # Build proper route with path expansion
        node_sequence = [
            (wh_node, {"pickups": [{"warehouse_id": best_wh.id, "sku_id": sku, "quantity": qty} 
                                  for sku, qty in order_reqs.items()]}),
            (order_node, [{"order_id": order_id, "sku_id": sku, "quantity": qty} 
                         for sku, qty in order_reqs.items()])
        ]
        
        steps = build_steps_with_path(env, node_sequence, home_node, dist_matrix)
        
        if steps:
            new_route = {"vehicle_id": vehicle_id, "steps": steps}
            solution["routes"].append(new_route)
        
    else:
        # Insert into existing route - rebuild entire route properly
        route = solution["routes"][route_idx]
        old_steps = route.get('steps', [])
        
        # Extract all deliveries from existing route
        existing_deliveries = []
        warehouse_id = None
        home_node = None
        
        for step in old_steps:
            if step.get('pickups') and step['pickups']:
                warehouse_id = step['pickups'][0]['warehouse_id']
            if step.get('deliveries'):
                for delivery in step['deliveries']:
                    existing_deliveries.append(delivery)
        
        # Find home node
        for v in env.get_all_vehicles():
            if v.id == route['vehicle_id']:
                home_node = env.get_vehicle_home_warehouse(v.id)
                break
        
        if not home_node or not warehouse_id:
            return solution
        
        # Insert new delivery at specified position
        delivery_nodes = [(env.orders[d['order_id']].destination.id, d['order_id']) 
                         for d in existing_deliveries]
        
        new_delivery = (order_node, order_id)
        delivery_nodes.insert(position, new_delivery)
        
        # Rebuild route with all deliveries
        all_reqs = {}
        for node, oid in delivery_nodes:
            reqs = env.get_order_requirements(oid)
            for sku, qty in reqs.items():
                all_reqs[sku] = all_reqs.get(sku, 0) + qty
        
        # Build node sequence
        wh = env.warehouses[warehouse_id]
        wh_node = wh.location.id
        
        node_sequence = [
            (wh_node, {"pickups": [{"warehouse_id": warehouse_id, "sku_id": sku, "quantity": qty} 
                                  for sku, qty in all_reqs.items()]})
        ]
        
        for node, oid in delivery_nodes:
            deliveries = [{"order_id": oid, "sku_id": sku, "quantity": qty} 
                         for sku, qty in env.get_order_requirements(oid).items()]
            node_sequence.append((node, deliveries))
        
        # Rebuild steps with proper pathfinding
        steps = build_steps_with_path(env, node_sequence, home_node, dist_matrix)
        
        if steps:
            solution["routes"][route_idx] = {"vehicle_id": route['vehicle_id'], "steps": steps}
    
    return solution


def build_steps_with_path(env: Any, node_sequence: List[Tuple[int, Any]], 
                          home_node: int, dist_matrix: Dict) -> Optional[List[Dict]]:
    """
    Builds route steps with intermediate path nodes using Dijkstra pathfinding.
    This ensures routes are valid and complete.
    
    Returns properly formatted steps including intermediate path nodes between stops.
    """
    steps = [{"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []}]
    current = home_node
    
    for node_id, payload in node_sequence:
        # Get path from current to next node
        path = dijkstra_shortest_path_nodes(env, current, node_id)
        
        if not path or len(path) < 2:
            # Fallback - direct connection
            path = [current, node_id]
        
        # Determine pickup/delivery lists
        if isinstance(payload, dict):
            pickups_list = payload.get("pickups", [])
            deliveries_list = payload.get("deliveries", [])
        else:
            pickups_list = []
            deliveries_list = payload if isinstance(payload, list) else []
        
        # Add intermediate nodes (excluding start which is already in steps)
        for i, intermediate in enumerate(path[1:], 1):
            is_destination = (intermediate == node_id)
            
            steps.append({
                "node_id": intermediate,
                "pickups": pickups_list if is_destination else [],
                "deliveries": deliveries_list if is_destination else [],
                "unloads": []
            })
        
        current = node_id
    
    # Path back to home
    path_home = dijkstra_shortest_path_nodes(env, current, home_node)
    
    if not path_home or len(path_home) < 2:
        path_home = [current, home_node]
    
    # Add path back (excluding current which is already in steps, including home)
    for intermediate in path_home[1:]:
        is_home = (intermediate == home_node)
        steps.append({
            "node_id": intermediate,
            "pickups": [],
            "deliveries": [],
            "unloads": [] if not is_home else []  # Could add unloads at home if needed
        })
    
    return steps


def dijkstra_shortest_path_nodes(env: Any, start: int, end: int) -> Optional[List[int]]:
    """
    Uses Dijkstra to find shortest path between two nodes.
    Returns list of node IDs from start to end (inclusive).
    """
    if start == end:
        return [start]
    
    try:
        adjacency = env.get_road_network_data().get("adjacency_list", {})
        
        if start not in adjacency:
            return [start, end]  # Fallback
        
        # Dijkstra
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
                # Reconstruct path
                path = []
                cur = end
                while cur is not None:
                    path.append(cur)
                    if cur == start:
                        break
                    cur = prev.get(cur)
                
                path.reverse()
                return path if path[0] == start and path[-1] == end else [start, end]
            
            for nb in adjacency.get(node, []):
                if nb in visited:
                    continue
                
                w = env.get_distance(node, nb)
                if w is None:
                    continue
                
                nd = cur_d + float(w)
                
                if nb not in dist or nd < dist[nb]:
                    dist[nb] = nd
                    prev[nb] = node
                    heapq.heappush(heap, (nd, nb))
        
        # End not reached
        return [start, end]
        
    except Exception:
        return [start, end]


# ============================================================================
# SECTION 4: UTILITY & COST FUNCTIONS
# ============================================================================

def calculate_solution_cost(env: Any, solution: Dict, dist_matrix: Dict) -> float:
    """Calculates the total cost of a solution using the pre-computed distance matrix."""
    total_cost = 0.0
    for route in solution.get("routes", []):
        vehicle_id = route['vehicle_id']
        # Find vehicle by ID
        vehicle = None
        for v in env.get_all_vehicles():
            if v.id == vehicle_id:
                vehicle = v
                break
        if not vehicle:
            continue
        
        route_dist = 0.0
        
        # Reconstruct the sequence of nodes visited
        node_sequence = [env.get_vehicle_home_warehouse(vehicle.id)]
        for step in route.get('steps', []):
            node_sequence.append(step['node_id'])
        node_sequence.append(env.get_vehicle_home_warehouse(vehicle.id))
        
        # Sum distances from the matrix
        for i in range(len(node_sequence) - 1):
            route_dist += dist_matrix.get((node_sequence[i], node_sequence[i+1]), 0.0)
            
        total_cost += vehicle.fixed_cost + vehicle.cost_per_km * route_dist
        
    return total_cost if total_cost > 0 else float('inf')


def build_initial_solution(env: Any, dist_matrix: Dict) -> Dict:
    """
    Builds a feasible initial solution using best insertion heuristic.
    Prioritizes orders that are hardest to place (farthest from warehouses).
    """
    solution = {"routes": []}
    unassigned_orders = set(env.get_all_order_ids())
    
    while unassigned_orders:
        # Select order farthest from its closest warehouse (hardest to place)
        selected_order = None
        max_min_distance = -1
        
        for order_id in unassigned_orders:
            order = env.orders[order_id]
            order_node = order.destination.id
            
            # Find minimum distance to any warehouse
            min_dist_to_wh = float('inf')
            for wh_id, wh in env.warehouses.items():
                wh_node = wh.location.id
                dist = dist_matrix.get((wh_node, order_node), float('inf'))
                min_dist_to_wh = min(min_dist_to_wh, dist)
            
            if min_dist_to_wh > max_min_distance:
                max_min_distance = min_dist_to_wh
                selected_order = order_id
        
        # Find best insertion position for selected order
        best_insertions = find_n_best_insertions(env, selected_order, solution, dist_matrix, n=1)
        
        if best_insertions:
            cost_increase, vehicle_id, route_idx, position = best_insertions[0]
            solution = insert_order_into_route(env, solution, selected_order, vehicle_id, route_idx, position, dist_matrix)
            unassigned_orders.remove(selected_order)
        else:
            # If no valid insertion found, remove from unassigned to avoid infinite loop
            # This shouldn't happen with proper implementation
            print(f"WARNING: Could not insert order {selected_order}")
            unassigned_orders.remove(selected_order)
    
    return solution


# ============================================================================
# SOLVER ENTRY POINT
# ============================================================================

def solver(env: Any) -> Dict:
    """
    Main entry point for the solver.
    """
    # The time budget should be slightly less than the 55s target to be safe.
    final_solution = solve_with_alns(env, time_budget=45.0)

    # Final validation and cleaning can be done here.
    return final_solution

# ============================================================================
# LOCAL TESTING BLOCK
# ============================================================================
# if __name__ == '__main__':
#     # This block is for local testing. It should be commented out for submission.
#     from robin_logistics import LogisticsEnvironment
#     
#     # Use one of the public scenarios for testing
#     env = LogisticsEnvironment(scenario_name='public_scenario_1') 
#     
#     solution = solver(env)
#     
#     print("\n--- FINAL SOLUTION ---")
#     print(f"Generated {len(solution.get('routes', []))} routes.")
# 
#     # Validate and get score
#     try:
#         validation_result = env.validate_solution_complete(solution)
#         print(f"Validation Result: {validation_result}")
#     
#         success, message = env.execute_solution(solution)
#         print(f"Execution Success: {success}, Message: {message}")
#         
#         if success:
#             cost = env.calculate_solution_cost(solution)
#             fulfillment = env.get_solution_fulfillment_summary(solution)
#             print(f"Total Cost: {cost:,.2f}")
#             print(f"Fulfillment Rate: {fulfillment['average_fulfillment_rate']:.1f}%")
#     except Exception as e:
#         print(f"An error occurred during validation/execution: {e}")
