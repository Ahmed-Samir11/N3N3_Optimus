"""
Optimized logistics solver with enhanced path finding.

This solution focuses on correctly implementing path finding for valid routes
that follow the road network.
"""
import heapq
from typing import Any, Dict, List, Set

def find_path(env, start, end):
    """Find a path from start to end using Dijkstra's algorithm with proper node handling."""
    if start == end:
        return [start]
        
    # Get the road network data
    road_data = env.get_road_network_data()
    adjacency_list = road_data.get("adjacency_list", {})
    
    # Convert to strings if they're not already
    start_str = str(start)
    end_str = str(end)
    
    # Check if nodes exist in adjacency list
    start_in_adj = start in adjacency_list
    start_str_in_adj = start_str in adjacency_list
    end_in_adj = end in adjacency_list
    end_str_in_adj = end_str in adjacency_list
    
    if not (start_in_adj or start_str_in_adj) or not (end_in_adj or end_str_in_adj):
        return [start, end]  # Fallback to direct connection
        
    # Initialize Dijkstra's algorithm
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    visited = set()
    
    # Use the correct key format for adjacency list lookups
    start_key = start_str if start_str_in_adj and not start_in_adj else start
        
    while pq:
        d, node = heapq.heappop(pq)
        
        # Convert to node key for lookup
        node_key = str(node) if str(node) in adjacency_list and node not in adjacency_list else node
            
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
            return path
            
        # Get neighbors
        neighbors = []
        if node_key in adjacency_list:
            neighbors.extend(adjacency_list[node_key])
            
        for neighbor in neighbors:
            if neighbor in visited:
                continue
                
            # Use uniform weight since we just need connectivity
            alt = d + 1
            if neighbor not in dist or alt < dist[neighbor]:
                dist[neighbor] = alt
                prev[neighbor] = node
                heapq.heappush(pq, (alt, neighbor))
    
    # No path found
    return [start, end]  # Fallback to direct connection

def build_expanded_route(env: Any, vehicle_id: str, compact_steps: List[Dict]) -> Dict:
    """Build a route with intermediate path nodes expanded inline."""
    if not compact_steps:
        return {"vehicle_id": vehicle_id, "steps": []}
    
    expanded_steps = []
    
    # Get vehicle home location
    try:
        # Get vehicle home warehouse/location
        try:
            home_wh_or_node = env.get_vehicle_home_warehouse(vehicle_id)
        except Exception:
            home_wh_or_node = None
            
        if home_wh_or_node is None:
            # If no home is specified, use the first step's node as home
            home_node = compact_steps[0]["node_id"]
        else:
            if home_wh_or_node in env.warehouses:
                try:
                    wh_obj = env.warehouses[home_wh_or_node]
                    home_node = getattr(wh_obj.location, "id", wh_obj.location)
                except Exception:
                    home_node = compact_steps[0]["node_id"]
            else:
                home_node = home_wh_or_node
    except Exception:
        # Fallback to first step node
        home_node = compact_steps[0]["node_id"]
    
    # Check if the route starts and ends at home
    starts_at_home = compact_steps[0]["node_id"] == home_node
    ends_at_home = compact_steps[-1]["node_id"] == home_node
    
    # Ensure route starts at home
    if not starts_at_home:
        home_step = {"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []}
        compact_steps.insert(0, home_step)
    
    # Ensure route ends at home
    if not ends_at_home:
        home_step = {"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []}
        compact_steps.append(home_step)
    
    # Add intermediate path nodes between steps
    for i in range(len(compact_steps)):
        current_node = compact_steps[i]["node_id"]
        
        # Add intermediate nodes if not the first step
        if i > 0:
            prev_node = compact_steps[i-1]["node_id"]
            if prev_node != current_node:
                path = find_path(env, prev_node, current_node)
                # Add intermediate nodes (skip first as it's already in previous step, skip last as it's current step)
                for intermediate in path[1:-1]:
                    expanded_steps.append({
                        "node_id": intermediate,
                        "pickups": [],
                        "deliveries": [],
                        "unloads": []
                    })
        
        # Add the actual step
        expanded_steps.append(compact_steps[i])
    
    return {"vehicle_id": vehicle_id, "steps": expanded_steps}

def solve(env):
    """Solve the logistics problem and create a solution."""
    # Get all data
    vehicles = list(env.get_all_vehicles())
    orders = list(env.get_all_order_ids())
    warehouses = env.warehouses
    
    print(f"Found {len(vehicles)} vehicles, {len(orders)} orders, {len(warehouses)} warehouses")
    
    # Track solution and used vehicles
    solution = {"routes": []}
    used_vehicles = set()
    
    # Greedy algorithm - try to assign each order to a vehicle
    for order_id in orders:
        # Skip if we've run out of vehicles
        if len(used_vehicles) >= len(vehicles):
            print(f"Ran out of vehicles after {len(solution['routes'])} routes")
            break
            
        # Get order requirements and location
        try:
            requirements = env.get_order_requirements(order_id)
            order_node = env.get_order_location(order_id)
        except Exception as e:
            print(f"Order {order_id} - failed to get requirements or location: {e}")
            continue
            
        if not requirements:
            print(f"Order {order_id} - no requirements")
            continue
            
        print(f"Processing order {order_id} at node {order_node} with {len(requirements)} SKUs")
            
        # Find closest warehouse (or just pick first if distances don't work)
        best_wh = None
        best_distance = float('inf')
        
        for wh_id, wh_obj in warehouses.items():
            try:
                wh_node = getattr(wh_obj.location, "id", wh_obj.location)
                distance = env.get_distance(wh_node, order_node)
                
                # Handle None distance (nodes not connected)
                if distance is None:
                    # If no distance available, just use the first warehouse
                    if best_wh is None:
                        best_wh = wh_id
                        best_distance = 0  # Unknown distance
                    continue
                
                if distance < best_distance:
                    best_distance = distance
                    best_wh = wh_id
            except Exception as e:
                print(f"  Error with warehouse {wh_id}: {e}")
                continue
                
        if not best_wh:
            print(f"  No warehouse found for order {order_id}")
            continue
            
        print(f"  Using warehouse: {best_wh}")
            
        # Find available vehicle
        vehicle = None
        for v in vehicles:
            if v.id not in used_vehicles:
                vehicle = v
                break
                
        if not vehicle:
            print(f"  No available vehicle for order {order_id}")
            continue
            
        print(f"  Using vehicle: {vehicle.id}")
            
        # Create the route
        try:
            print(f"Creating route for order {order_id} with vehicle {vehicle.id} from warehouse {best_wh}")
            # Get nodes
            wh_node = getattr(warehouses[best_wh].location, "id", warehouses[best_wh].location)
            print(f"  Warehouse node: {wh_node}")
            
            # Get vehicle home
            try:
                home_wh_or_node = env.get_vehicle_home_warehouse(vehicle.id)
            except Exception:
                home_wh_or_node = None
                
            if home_wh_or_node is None:
                home_node = wh_node
            else:
                if home_wh_or_node in warehouses:
                    try:
                        home_node = getattr(warehouses[home_wh_or_node].location, "id", warehouses[home_wh_or_node].location)
                    except Exception:
                        home_node = wh_node
                else:
                    home_node = home_wh_or_node
            
            # Build steps
            compact_steps = []
            
            # Always start at home
            compact_steps.append({
                "node_id": home_node, 
                "pickups": [], 
                "deliveries": [], 
                "unloads": []
            })
            
            # Go to warehouse if different from home
            if home_node != wh_node:
                compact_steps.append({
                    "node_id": wh_node, 
                    "pickups": [
                        {"warehouse_id": best_wh, "sku_id": sku, "quantity": qty} 
                        for sku, qty in requirements.items()
                    ],
                    "deliveries": [], 
                    "unloads": []
                })
            else:
                # If home is warehouse, add pickups to home step
                compact_steps[0]["pickups"] = [
                    {"warehouse_id": best_wh, "sku_id": sku, "quantity": qty} 
                    for sku, qty in requirements.items()
                ]
            
            # Go to order location
            compact_steps.append({
                "node_id": order_node, 
                "pickups": [], 
                "deliveries": [
                    {"order_id": order_id, "sku_id": sku, "quantity": qty} 
                    for sku, qty in requirements.items()
                ], 
                "unloads": []
            })
            
            # Return to home
            compact_steps.append({
                "node_id": home_node, 
                "pickups": [], 
                "deliveries": [], 
                "unloads": []
            })
            
            # Build expanded route with paths
            route = build_expanded_route(env, vehicle.id, compact_steps)
            
            # Validate route
            valid = False
            try:
                validation_result = env.validate_solution_complete({"routes": [route]})
                
                if isinstance(validation_result, dict):
                    valid = validation_result.get("valid", False)
                    if not valid:
                        print(f"Order {order_id} route INVALID: {validation_result.get('errors', 'Unknown error')}")
                elif isinstance(validation_result, tuple):
                    valid = validation_result[0]
                    if not valid:
                        print(f"Order {order_id} route INVALID: {validation_result[1]}")
                else:
                    valid = bool(validation_result)
                    if not valid:
                        print(f"Order {order_id} route INVALID")
            except Exception as e:
                valid = False
                print(f"Order {order_id} validation ERROR: {e}")
                
            if valid:
                solution["routes"].append(route)
                used_vehicles.add(vehicle.id)
                print(f"Order {order_id} route VALID - added to solution")
                
        except Exception as e:
            print(f"Order {order_id} route creation ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return solution

# Alias for compatibility with test framework
def my_solver(env):
    """Wrapper function for test framework compatibility."""
    return solve(env)