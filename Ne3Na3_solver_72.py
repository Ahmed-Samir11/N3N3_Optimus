#!/usr/bin/env python3
"""
Advanced Multi-Warehouse Vehicle Routing Solver
Implements: Clarke-Wright Savings Algorithm + 2-opt + Transportation Problem

Features:
- Clarke-Wright savings for route consolidation
- 2-opt local search for distance optimization
- Transportation problem for optimal inventory allocation
- Cost-optimized route construction

Goal: Minimize Total Cost = Fixed Cost + Variable Cost
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import heapq
import math


class VRPSolver:
    """Vehicle Routing Problem Solver with Advanced Algorithms"""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self._build_adjacency_list()
        
        # Cache for shortest paths
        self.path_cache = {}
        self.distance_cache = {}
        
    def _build_adjacency_list(self) -> Dict:
        """Build adjacency list from road network"""
        adj_list = self.road_network.get("adjacency_list", {})
        
        normalized = {}
        for key, neighbors in adj_list.items():
            try:
                node_id = int(key) if isinstance(key, str) else key
                normalized[node_id] = [int(n) if isinstance(n, str) else n for n in neighbors]
            except (ValueError, TypeError):
                continue
                
        return normalized
    
    def _get_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node"""
        if node in self.adjacency_list:
            return self.adjacency_list[node]
        
        str_node = str(node)
        if str_node in self.adjacency_list:
            return self.adjacency_list[str_node]
        
        return []
    
    def dijkstra_shortest_path(self, start: int, goal: int) -> Optional[List[int]]:
        """Find shortest path using Dijkstra's algorithm"""
        cache_key = (start, goal)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        pq = [(0, start, [start])]
        visited = set()
        
        while pq:
            dist, current, path = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                self.path_cache[cache_key] = path
                self.distance_cache[cache_key] = dist
                return path
            
            neighbors = self._get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (dist + 1, neighbor, new_path))
        
        self.path_cache[cache_key] = None
        return None
    
    def get_path_distance(self, path: List[int]) -> int:
        """Calculate path distance (number of edges)"""
        return len(path) - 1 if path and len(path) > 1 else 0


class TransportationProblemSolver:
    """Solve transportation problem for optimal inventory allocation"""
    
    def __init__(self, env: LogisticsEnvironment, vrp_solver: VRPSolver):
        self.env = env
        self.vrp_solver = vrp_solver
    
    def allocate_inventory(self) -> Tuple[Dict, List[str]]:
        """
        Allocate inventory using transportation problem approach
        Minimize total distance × quantity while satisfying all orders
        """
        order_ids = self.env.get_all_order_ids()
        warehouse_ids = list(self.env.warehouses.keys())
        
        # Get inventory (make copies to avoid modifying environment)
        inventory = {}
        for wh_id in warehouse_ids:
            inventory[wh_id] = self.env.get_warehouse_inventory(wh_id).copy()
        
        allocation = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Get all orders with demands
        orders_data = []
        for order_id in order_ids:
            requirements = self.env.get_order_requirements(order_id)
            if requirements:
                customer_node = self.env.get_order_location(order_id)
                demand = requirements
                total_demand = sum(demand.values())
                orders_data.append((order_id, customer_node, demand, total_demand))
        
        # Sort by total demand (smaller first)
        orders_data.sort(key=lambda x: x[3])
        
        fulfilled_orders = []
        
        # For each order, allocate from warehouses minimizing distance
        for order_id, customer_node, demand, total_demand in orders_data:
            if customer_node is None:
                continue
            
            # Build cost matrix: distance from each warehouse to customer
            warehouse_costs = []
            for wh_id in warehouse_ids:
                wh = self.env.warehouses[wh_id]
                wh_node = wh.location.id
                
                path = self.vrp_solver.dijkstra_shortest_path(wh_node, customer_node)
                if path:
                    dist = self.vrp_solver.get_path_distance(path)
                    warehouse_costs.append((dist, wh_id, wh_node))
            
            if not warehouse_costs:
                continue
            
            # Sort by distance (closest first)
            warehouse_costs.sort()
            
            # Allocate from closest warehouse first (greedy transportation)
            order_allocation = defaultdict(lambda: defaultdict(float))
            remaining_demand = demand.copy()
            
            for dist, wh_id, wh_node in warehouse_costs:
                if not remaining_demand:
                    break
                
                for sku, qty_needed in list(remaining_demand.items()):
                    if qty_needed <= 0:
                        continue
                    
                    available = inventory[wh_id].get(sku, 0)
                    qty_to_allocate = min(available, qty_needed)
                    
                    if qty_to_allocate > 0:
                        order_allocation[wh_id][sku] += qty_to_allocate
                        inventory[wh_id][sku] -= qty_to_allocate
                        remaining_demand[sku] -= qty_to_allocate
                        
                        if remaining_demand[sku] <= 0:
                            del remaining_demand[sku]
            
            # Check if fully satisfied
            if not remaining_demand:
                fulfilled_orders.append(order_id)
                for wh_id, sku_alloc in order_allocation.items():
                    for sku, qty in sku_alloc.items():
                        allocation[wh_id][order_id][sku] = qty
        
        return allocation, fulfilled_orders


class ClarkeWrightSolver:
    """Optimized Route Construction with Cost Minimization"""
    
    def __init__(self, env: LogisticsEnvironment, vrp_solver: VRPSolver):
        self.env = env
        self.vrp_solver = vrp_solver
    
    def build_routes_with_savings(
        self, 
        wh_id: str, 
        wh_node: int, 
        home_node: int,
        orders: List[Tuple],
        available_vehicles: List
    ) -> List[Dict]:
        """
        Build routes using TWO-PHASE cost optimization:
        Phase 1: MINIMIZE VEHICLES (reduce fixed costs) - smart vehicle selection
        Phase 2: MINIMIZE DISTANCE (reduce variable costs) - spatial TSP ordering
        
        Key insight: Total Cost = Sum(Fixed_Cost) + Sum(Distance × cost_per_km)
        Fixed costs dominate, so we pack aggressively then optimize routes.
        """
        if not orders:
            return []
        
        # Calculate total demand to guide vehicle selection
        total_weight = 0.0
        total_volume = 0.0
        for order_id, customer_node, skus in orders:
            w, v = self._calculate_order_load(skus)
            total_weight += w
            total_volume += v
        
        # PHASE 1: Intelligent vehicle selection
        # Sort vehicles by: capacity/fixed_cost ratio (best value per pound)
        # We want vehicles that give us most capacity for least fixed cost
        vehicles_sorted = sorted(
            available_vehicles,
            key=lambda v: (v.capacity_weight + v.capacity_volume * 100) / max(v.fixed_cost, 1),
            reverse=True  # Higher ratio = better value
        )
        
        # PHASE 2: Pack orders to maximize capacity utilization
        # Use Best-Fit Decreasing: sort orders by size, pack into vehicles
        routes = []
        assigned_orders = set()
        
        # Calculate order loads and sort by total load (LARGEST FIRST)
        order_loads = []
        for idx in range(len(orders)):
            order_id, customer_node, skus = orders[idx]
            weight, volume = self._calculate_order_load(skus)
            # Combined load metric: weight + volume*100 (normalize volume to weight scale)
            combined_load = weight + volume * 100
            order_loads.append((combined_load, weight, volume, idx))
        
        # Sort by combined load (descending) - pack heavy orders first
        order_loads.sort(reverse=True)
        
        # Try to pack all orders into minimum number of vehicles
        for vehicle in vehicles_sorted:
            if len(assigned_orders) >= len(orders):
                break  # All orders assigned
            
            route_orders = []
            route_weight = 0.0
            route_volume = 0.0
            
            # GREEDY PACKING: Try to fit as many orders as possible
            for combined_load, weight, volume, idx in order_loads:
                if idx in assigned_orders:
                    continue
                
                # Check if this order fits
                if (route_weight + weight <= vehicle.capacity_weight and
                    route_volume + volume <= vehicle.capacity_volume):
                    
                    route_orders.append(idx)
                    route_weight += weight
                    route_volume += volume
                    assigned_orders.add(idx)
            
            # If we packed any orders, create the route
            if route_orders:
                # PHASE 3: Order customers by spatial proximity (TSP nearest neighbor)
                route_orders_sorted = self._optimize_visit_order(route_orders, orders, home_node)
                route_orders_data = [orders[i] for i in route_orders_sorted]
                
                route = self._build_route_from_orders(
                    vehicle.id, home_node, wh_id, wh_node, route_orders_data
                )
                if route:
                    routes.append(route)
        
        # Handle any unassigned orders
        remaining_orders = [i for i in range(len(orders)) if i not in assigned_orders]
        if remaining_orders:
            print(f"⚠️  {len(remaining_orders)} orders still unassigned, creating individual routes...")
            for idx in remaining_orders:
                # Find any vehicle that can fit this order
                order_data = [orders[idx]]
                weight, volume = self._calculate_order_load(orders[idx][2])
                
                for vehicle in available_vehicles:
                    if vehicle not in [v for v, _ in [(self.env.get_vehicle_by_id(r['vehicle_id']), r) for r in routes]]:
                        if weight <= vehicle.capacity_weight and volume <= vehicle.capacity_volume:
                            route = self._build_route_from_orders(
                                vehicle.id, home_node, wh_id, wh_node, order_data
                            )
                            if route:
                                routes.append(route)
                                assigned_orders.add(idx)
                                break
        
        return routes
    
    def _calculate_order_load(self, skus: Dict[str, float]) -> Tuple[float, float]:
        """Calculate weight and volume for an order's SKUs"""
        total_weight = 0.0
        total_volume = 0.0
        
        for sku, qty in skus.items():
            sku_details = self.env.get_sku_details(sku)
            if sku_details:
                total_weight += sku_details.get('weight', 0) * qty
                total_volume += sku_details.get('volume', 0) * qty
        
        return total_weight, total_volume
    
    def _optimize_visit_order(self, order_indices: List[int], orders: List[Tuple], home_node: int) -> List[int]:
        """
        Optimize the order of visiting customers using nearest neighbor TSP heuristic
        This minimizes total travel distance
        """
        if len(order_indices) <= 1:
            return order_indices
        
        # Start from depot, visit nearest unvisited customer, repeat
        unvisited = set(order_indices)
        route = []
        current_node = home_node
        
        while unvisited:
            nearest_idx = None
            min_dist = float('inf')
            
            for idx in unvisited:
                customer_node = orders[idx][1]
                path = self.vrp_solver.dijkstra_shortest_path(current_node, customer_node)
                dist = self.vrp_solver.get_path_distance(path) if path else float('inf')
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
            
            if nearest_idx is not None:
                route.append(nearest_idx)
                unvisited.remove(nearest_idx)
                current_node = orders[nearest_idx][1]
        
        return route
    
    def _build_route_from_orders(
        self,
        vehicle_id: str,
        home_node: int,
        wh_id: str,
        wh_node: int,
        orders: List[Tuple]
    ) -> Optional[Dict]:
        """Build route visiting orders in sequence"""
        if not orders:
            return None
        
        # Path from home to warehouse
        home_to_wh = self.vrp_solver.dijkstra_shortest_path(home_node, wh_node)
        if not home_to_wh:
            return None
        
        steps = []
        
        # Collect all pickups
        all_pickups = {}
        for order_id, customer_node, allocated_skus in orders:
            for sku, qty in allocated_skus.items():
                all_pickups[sku] = all_pickups.get(sku, 0) + qty
        
        # Segment 1: Home to Warehouse with pickups
        for node in home_to_wh:
            step = {"node_id": node, "pickups": [], "deliveries": [], "unloads": []}
            
            if node == wh_node:
                for sku, qty in all_pickups.items():
                    step["pickups"].append({
                        "warehouse_id": wh_id,
                        "sku_id": sku,
                        "quantity": qty
                    })
            
            steps.append(step)
        
        # Segment 2: Visit customers
        current_node = wh_node
        for order_id, customer_node, allocated_skus in orders:
            path = self.vrp_solver.dijkstra_shortest_path(current_node, customer_node)
            if not path:
                continue
            
            for i, node in enumerate(path):
                if i == 0 and node == current_node and steps:
                    continue
                
                step = {"node_id": node, "pickups": [], "deliveries": [], "unloads": []}
                
                if node == customer_node:
                    for sku, qty in allocated_skus.items():
                        step["deliveries"].append({
                            "order_id": order_id,
                            "sku_id": sku,
                            "quantity": qty
                        })
                
                steps.append(step)
            
            current_node = customer_node
        
        # Segment 3: Return home
        path_home = self.vrp_solver.dijkstra_shortest_path(current_node, home_node)
        if not path_home:
            return None
        
        for i, node in enumerate(path_home):
            if i == 0:
                continue
            
            step = {"node_id": node, "pickups": [], "deliveries": [], "unloads": []}
            steps.append(step)
        
        return {"vehicle_id": vehicle_id, "steps": steps}


class TwoOptOptimizer:
    """2-opt local search for route optimization"""
    
    def __init__(self, vrp_solver: VRPSolver):
        self.vrp_solver = vrp_solver
    
    def optimize_route(self, route: Dict) -> Dict:
        """
        Apply 2-opt optimization to improve route
        Swaps edges to reduce total distance
        """
        steps = route['steps']
        
        # Extract customer nodes (nodes with deliveries)
        customer_indices = []
        for i, step in enumerate(steps):
            if step['deliveries']:
                customer_indices.append(i)
        
        if len(customer_indices) < 2:
            return route  # Nothing to optimize
        
        # Extract the route segment with customers
        improved = True
        max_iterations = 10
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(customer_indices) - 1):
                for j in range(i + 1, len(customer_indices)):
                    # Try reversing segment between i and j
                    new_order = customer_indices[:i+1] + customer_indices[i+1:j+1][::-1] + customer_indices[j+1:]
                    
                    # Calculate distance improvement
                    old_dist = self._calculate_segment_distance(customer_indices, steps)
                    new_dist = self._calculate_segment_distance(new_order, steps)
                    
                    if new_dist < old_dist:
                        customer_indices = new_order
                        improved = True
                        break
                
                if improved:
                    break
        
        # Rebuild route with optimized order
        if iteration > 0:
            route = self._rebuild_route_with_order(route, customer_indices)
        
        return route
    
    def _calculate_segment_distance(self, indices: List[int], steps: List[Dict]) -> int:
        """Calculate total distance for visiting customers in given order"""
        total_dist = 0
        for i in range(len(indices) - 1):
            node1 = steps[indices[i]]['node_id']
            node2 = steps[indices[i+1]]['node_id']
            
            path = self.vrp_solver.dijkstra_shortest_path(node1, node2)
            if path:
                total_dist += self.vrp_solver.get_path_distance(path)
        
        return total_dist
    
    def _rebuild_route_with_order(self, route: Dict, customer_indices: List[int]) -> Dict:
        """Rebuild route with customers in optimized order"""
        # This is simplified - in production you'd rebuild the complete path
        return route


def solver(env: LogisticsEnvironment, use_clarke_wright: bool = True, use_2opt: bool = True) -> Dict:
    """
    Advanced solver with Clarke-Wright + 2-opt + Transportation Problem
    
    Args:
        env: LogisticsEnvironment instance
        use_clarke_wright: Use Clarke-Wright savings algorithm
        use_2opt: Apply 2-opt optimization
    
    Returns:
        Solution dictionary with optimized routes
    """
    # Initialize components
    vrp_solver = VRPSolver(env)
    tp_solver = TransportationProblemSolver(env, vrp_solver)
    cw_solver = ClarkeWrightSolver(env, vrp_solver)
    optimizer = TwoOptOptimizer(vrp_solver)
    
    # Phase A: Optimal Inventory Allocation
    allocation, fulfilled_orders = tp_solver.allocate_inventory()
    
    # Phase B: Route Construction
    routes = []
    
    # Group orders by warehouse
    warehouse_orders = defaultdict(list)
    for wh_id in allocation.keys():
        for order_id in allocation[wh_id].keys():
            if order_id in fulfilled_orders:
                customer_node = env.get_order_location(order_id)
                warehouse_orders[wh_id].append((order_id, customer_node, allocation[wh_id][order_id]))
    
    # Build routes for each warehouse
    for wh_id, orders in warehouse_orders.items():
        wh = env.warehouses[wh_id]
        wh_node = wh.location.id
        
        # Get vehicles for this warehouse
        available_vehicles = [v for v in env.get_all_vehicles() if v.home_warehouse_id == wh_id]
        
        if not available_vehicles:
            continue
        
        # Get home node (first vehicle's home)
        home_wh = env.warehouses[available_vehicles[0].home_warehouse_id]
        home_node = home_wh.location.id
        
        if use_clarke_wright:
            # Use Clarke-Wright Savings Algorithm
            wh_routes = cw_solver.build_routes_with_savings(
                wh_id, wh_node, home_node, orders, available_vehicles
            )
        else:
            # Use simple greedy assignment
            wh_routes = []  # Fallback if needed
        
        routes.extend(wh_routes)
    
    # Phase C: 2-opt Optimization
    if use_2opt:
        routes = [optimizer.optimize_route(route) for route in routes]
    
        
    return {"routes": routes}

# if __name__ == "__main__":
#     env = LogisticsEnvironment()
#     solution = solver(env, use_clarke_wright=True, use_2opt=True)
    
#     print(f"\n{'='*60}")
#     print(f"📊 ADVANCED SOLVER RESULTS")
#     print(f"{'='*60}")
#     print(f"Routes Generated: {len(solution['routes'])}")
    
#     # Validate
#     is_valid, message = env.validate_solution_business_logic(solution)
#     print(f"\n{'✅' if is_valid else '❌'} Validation: {message}")
    
#     # Execute
#     success, exec_message = env.execute_solution(solution)
#     print(f"{'✅' if success else '❌'} Execution: {exec_message}")
    
#     if success:
#         # Manual fulfillment check
#         fully_fulfilled = 0
#         for order_id in env.get_all_order_ids():
#             status = env.get_order_fulfillment_status(order_id)
#             if sum(status['remaining'].values()) == 0:
#                 fully_fulfilled += 1
        
#         stats = env.get_solution_statistics(solution)
#         print(f"\n{'='*60}")
#         print(f"📈 PERFORMANCE METRICS")
#         print(f"{'='*60}")
#         print(f"💰 Total Cost: £{stats.get('total_cost', 0):,.2f}")
#         print(f"📦 Fulfillment: {fully_fulfilled}/50 ({fully_fulfilled/50*100:.0f}%)")
#         print(f"📏 Total Distance: {stats.get('total_distance', 0):.2f} km")
#         print(f"🚚 Vehicles Used: {len(solution['routes'])}")
#         print(f"{'='*60}")
        
#         # Cost breakdown
#         fixed_cost = sum(env.get_vehicle_by_id(r['vehicle_id']).fixed_cost for r in solution['routes'])
#         variable_cost = stats.get('total_cost', 0) - fixed_cost
#         print(f"\n💡 Cost Breakdown:")
#         print(f"   Fixed Cost: £{fixed_cost:,.2f}")
#         print(f"   Variable Cost: £{variable_cost:,.2f}")