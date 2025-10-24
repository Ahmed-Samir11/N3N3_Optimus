#!/usr/bin/env python3
"""
Multi-Warehouse Vehicle Routing Solver with Inventory Constraints
Based on Mathematical Model: Minimize S = C_you + C_bench × (100 - Fulfillment%)

Implementation Strategy:
Phase A: Inventory Allocation - Maximize full order fulfillment
Phase B: Route Construction - Clarke-Wright savings algorithm  
Phase C: Local Search - 2-opt optimization
Phase D: Output - Valid routes with complete paths

KEY FIX: Uses correct API attributes:
- vehicle.capacity_weight / vehicle.capacity_volume (not max_weight/max_volume)
- vehicle.home_warehouse_id (to get warehouse, then warehouse.location.id for node)
- env.warehouses dict is directly accessible
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import heapq
from itertools import combinations


class VRPSolver:
    """Vehicle Routing Problem Solver with Multi-Warehouse Support"""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self._build_adjacency_list()
        
        # Cache for shortest paths
        self.path_cache = {}
        self.distance_cache = {}
        
    def _build_adjacency_list(self) -> Dict:
        """Build adjacency list from road network, handling both string and int keys"""
        adj_list = self.road_network.get("adjacency_list", {})
        
        # Normalize keys to integers for consistent access
        normalized = {}
        for key, neighbors in adj_list.items():
            try:
                node_id = int(key) if isinstance(key, str) else key
                # Normalize neighbor IDs too
                normalized[node_id] = [int(n) if isinstance(n, str) else n for n in neighbors]
            except (ValueError, TypeError):
                continue
                
        return normalized
    
    def _get_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node, trying multiple key formats"""
        # Try direct access first
        if node in self.adjacency_list:
            return self.adjacency_list[node]
        
        # Try string key
        str_node = str(node)
        if str_node in self.adjacency_list:
            return self.adjacency_list[str_node]
        
        return []
    
    def dijkstra_shortest_path(self, start: int, goal: int) -> Optional[List[int]]:
        """
        Find shortest path using Dijkstra's algorithm
        Returns list of node IDs from start to goal (including both endpoints)
        """
        cache_key = (start, goal)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Priority queue: (distance, current_node, path)
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
        
        # No path found
        self.path_cache[cache_key] = None
        return None
    
    def get_path_distance(self, path: List[int]) -> int:
        """Calculate path distance (number of edges)"""
        return len(path) - 1 if path and len(path) > 1 else 0


class InventoryAllocator:
    """Phase A: Allocate inventory to maximize full order fulfillment"""
    
    def __init__(self, env: LogisticsEnvironment, solver: VRPSolver):
        self.env = env
        self.solver = solver
        
    def allocate_inventory(self) -> Tuple[Dict, List[str]]:
        """
        Allocate SKUs from warehouses to orders
        Returns: (allocation dict, fulfilled_orders list)
        
        Strategy: Greedy allocation prioritizing smaller orders first
        """        
        order_ids = self.env.get_all_order_ids()
        
        # Get warehouses from env.warehouses dict
        warehouse_ids = list(self.env.warehouses.keys())
        
        # Track remaining inventory per warehouse
        inventory = {}
        for wh_id in warehouse_ids:
            inventory[wh_id] = self.env.get_warehouse_inventory(wh_id).copy()
        
        # Allocation result
        allocation = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Sort orders by total demand (smaller first for higher fulfillment rate)
        orders_with_demand = []
        for order_id in order_ids:
            order_requirements = self.env.get_order_requirements(order_id)
            if not order_requirements:
                continue
            
            demand = order_requirements
            total_demand = sum(demand.values())
            orders_with_demand.append((order_id, total_demand, demand))

        orders_with_demand.sort(key=lambda x: x[1])
                
        # Greedy allocation
        fulfilled_orders = []
        partially_fulfilled = []
        failed_orders = []
                
        for idx, (order_id, total_demand, demand) in enumerate(orders_with_demand):
            customer_node = self.env.get_order_location(order_id)
            if customer_node is None:
                failed_orders.append((order_id, "No customer location"))
                continue
                        
            # Try to fulfill from closest warehouse first
            warehouse_distances = []
            for wh_id in warehouse_ids:
                wh = self.env.warehouses[wh_id]
                wh_node = wh.location.id
                
                # Calculate distance from warehouse to customer
                path = self.solver.dijkstra_shortest_path(wh_node, customer_node)
                if path:
                    dist = self.solver.get_path_distance(path)
                    warehouse_distances.append((dist, wh_id, wh_node))
            
            if not warehouse_distances:
                failed_orders.append((order_id, "No reachable warehouses"))
                continue
            
            warehouse_distances.sort()  # Closest first

            # Try to fulfill entire order from warehouses
            order_allocation = defaultdict(lambda: defaultdict(float))
            remaining_demand = demand.copy()
            
            for _, wh_id, wh_node in warehouse_distances:
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
            
            # Check if order is fully satisfied
            if not remaining_demand:
                fulfilled_orders.append(order_id)
                for wh_id, sku_alloc in order_allocation.items():
                    for sku, qty in sku_alloc.items():
                        allocation[wh_id][order_id][sku] = qty
            else:
                partially_fulfilled.append((order_id, remaining_demand))

        return allocation, fulfilled_orders


class RouteBuilder:
    """Phase B: Build delivery routes"""
    
    def __init__(self, env: LogisticsEnvironment, solver: VRPSolver):
        self.env = env
        self.solver = solver
        
    def build_routes(self, allocation: Dict, fulfilled_orders: List[str]) -> List[Dict]:
        """
        Build routes for fulfilled orders grouped by warehouse
        Returns list of route dictionaries
        """
        routes = []
        
        # Group orders by warehouse
        warehouse_orders = defaultdict(list)
        for wh_id in allocation.keys():
            for order_id in allocation[wh_id].keys():
                if order_id in fulfilled_orders:
                    customer_node = self.env.get_order_location(order_id)
                    warehouse_orders[wh_id].append((order_id, customer_node, allocation[wh_id][order_id]))
        
        # Build routes for each warehouse
        for wh_id, orders in warehouse_orders.items():
            wh = self.env.warehouses[wh_id]
            wh_node = wh.location.id
            
            # Get available vehicles for this warehouse
            available_vehicles = [v for v in self.env.get_all_vehicles() if v.home_warehouse_id == wh_id]
            
            # Simple greedy: assign one order per vehicle
            vehicle_routes = self._assign_orders_to_vehicles(
                wh_id, wh_node, orders, available_vehicles
            )
            
            routes.extend(vehicle_routes)
        
        return routes
    
    def _assign_orders_to_vehicles(
        self, 
        wh_id: str, 
        wh_node: int, 
        orders: List[Tuple], 
        available_vehicles: List
    ) -> List[Dict]:
        """Assign multiple orders to vehicles respecting capacity constraints"""
        routes = []
        
        # Track which orders are assigned and current load per vehicle
        unassigned_orders = list(orders)
        vehicle_assignments = {v.id: [] for v in available_vehicles}
        vehicle_loads = {v.id: {'weight': 0, 'volume': 0} for v in available_vehicles}
        
        # Greedy bin packing: assign orders to vehicles
        for order_id, customer_node, allocated_skus in unassigned_orders:
            # Calculate order weight and volume
            order_weight, order_volume = self._calculate_order_size(allocated_skus)
            
            # Try to fit in an existing vehicle
            assigned = False
            for vehicle in available_vehicles:
                current_weight = vehicle_loads[vehicle.id]['weight']
                current_volume = vehicle_loads[vehicle.id]['volume']
                
                if (current_weight + order_weight <= vehicle.capacity_weight and
                    current_volume + order_volume <= vehicle.capacity_volume):
                    # Fits! Assign to this vehicle
                    vehicle_assignments[vehicle.id].append((order_id, customer_node, allocated_skus))
                    vehicle_loads[vehicle.id]['weight'] += order_weight
                    vehicle_loads[vehicle.id]['volume'] += order_volume
                    assigned = True
                    break
            
            if not assigned:
                print(f"⚠️  Order {order_id} could not fit in any vehicle (weight={order_weight:.1f}, volume={order_volume:.3f})")
        
        # Build routes for vehicles with assignments
        for vehicle in available_vehicles:
            if vehicle_assignments[vehicle.id]:
                # Get vehicle home node
                home_wh = self.env.warehouses[vehicle.home_warehouse_id]
                home_node = home_wh.location.id
                
                # Build multi-order route
                route = self._build_multi_order_route(
                    vehicle.id, home_node, wh_id, wh_node,
                    vehicle_assignments[vehicle.id]
                )
                
                if route:
                    routes.append(route)
        
        return routes
    
    def _calculate_order_size(self, allocated_skus: Dict) -> Tuple[float, float]:
        """Calculate total weight and volume for an order"""
        total_weight = 0.0
        total_volume = 0.0
        
        for sku, qty in allocated_skus.items():
            sku_details = self.env.get_sku_details(sku)
            if sku_details:
                total_weight += sku_details.get('weight', 0) * qty
                total_volume += sku_details.get('volume', 0) * qty
        
        return total_weight, total_volume
    
    def _can_fit_order(self, vehicle, allocated_skus: Dict) -> bool:
        """Check if order fits in vehicle capacity"""
        total_weight = 0
        total_volume = 0
        
        for sku, qty in allocated_skus.items():
            sku_details = self.env.get_sku_details(sku)
            if sku_details:
                total_weight += sku_details.get('weight', 0) * qty
                total_volume += sku_details.get('volume', 0) * qty
        
        return (total_weight <= vehicle.capacity_weight and
                total_volume <= vehicle.capacity_volume)
    
    def _build_multi_order_route(
        self,
        vehicle_id: str,
        home_node: int,
        wh_id: str,
        wh_node: int,
        orders: List[Tuple[str, int, Dict]]
    ) -> Optional[Dict]:
        """
        Build a route with multiple orders
        Route: Home → Warehouse → Customer1 → Customer2 → ... → Home
        """
        if not orders:
            return None
        
        # Start with path from home to warehouse
        home_to_wh = self.solver.dijkstra_shortest_path(home_node, wh_node)
        if not home_to_wh:
            return None
        
        steps = []
        
        # Segment 1: Home to Warehouse with all pickups
        all_pickups = {}
        for order_id, customer_node, allocated_skus in orders:
            for sku, qty in allocated_skus.items():
                all_pickups[sku] = all_pickups.get(sku, 0) + qty
        
        for i, node in enumerate(home_to_wh):
            step = {"node_id": node, "pickups": [], "deliveries": [], "unloads": []}
            
            # Pickup at warehouse
            if node == wh_node:
                for sku, qty in all_pickups.items():
                    step["pickups"].append({
                        "warehouse_id": wh_id,
                        "sku_id": sku,
                        "quantity": qty
                    })
            
            steps.append(step)
        
        # Segment 2: Visit each customer in order
        current_node = wh_node
        for order_id, customer_node, allocated_skus in orders:
            # Path from current to customer
            path_to_customer = self.solver.dijkstra_shortest_path(current_node, customer_node)
            if not path_to_customer:
                continue
            
            # Add nodes (skip first if it's the current node)
            for i, node in enumerate(path_to_customer):
                if i == 0 and node == current_node and steps:
                    # Already added this node
                    continue
                
                step = {"node_id": node, "pickups": [], "deliveries": [], "unloads": []}
                
                # Delivery at customer
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
        path_to_home = self.solver.dijkstra_shortest_path(current_node, home_node)
        if not path_to_home:
            return None
        
        for i, node in enumerate(path_to_home):
            if i == 0:  # Skip first (already added)
                continue
            
            step = {"node_id": node, "pickups": [], "deliveries": [], "unloads": []}
            steps.append(step)
        
        return {
            "vehicle_id": vehicle_id,
            "steps": steps
        }
    
    def _build_single_order_route(
        self,
        vehicle_id: str,
        home_node: int,
        wh_id: str,
        wh_node: int,
        order_id: str,
        customer_node: int,
        allocated_skus: Dict[str, float]
    ) -> Optional[Dict]:
        """
        Build a complete route with ALL intermediate nodes
        Critical: Include every node in the path for distance calculation
        """
        # Get complete paths for each segment
        home_to_wh = self.solver.dijkstra_shortest_path(home_node, wh_node)
        wh_to_customer = self.solver.dijkstra_shortest_path(wh_node, customer_node)
        customer_to_home = self.solver.dijkstra_shortest_path(customer_node, home_node)
        
        if not (home_to_wh and wh_to_customer and customer_to_home):
            return None
        
        # Build steps with ALL nodes in the complete path
        steps = []
        
        # Segment 1: Home to Warehouse (all nodes, pickup at warehouse)
        for i, node in enumerate(home_to_wh):
            step = {"node_id": node, "pickups": [], "deliveries": [], "unloads": []}
            
            # Pickup at warehouse
            if node == wh_node:
                for sku, qty in allocated_skus.items():
                    step["pickups"].append({
                        "warehouse_id": wh_id,
                        "sku_id": sku,
                        "quantity": qty
                    })
            
            steps.append(step)
        
        # Segment 2: Warehouse to Customer (all nodes except first, delivery at customer)
        for i, node in enumerate(wh_to_customer):
            if i == 0:  # Skip warehouse (already added)
                continue
            
            step = {"node_id": node, "pickups": [], "deliveries": [], "unloads": []}
            
            # Delivery at customer
            if node == customer_node:
                for sku, qty in allocated_skus.items():
                    step["deliveries"].append({
                        "order_id": order_id,
                        "sku_id": sku,
                        "quantity": qty
                    })
            
            steps.append(step)
        
        # Segment 3: Customer to Home (all nodes except first)
        for i, node in enumerate(customer_to_home):
            if i == 0:  # Skip customer (already added)
                continue
            
            step = {"node_id": node, "pickups": [], "deliveries": [], "unloads": []}
            steps.append(step)
        
        return {
            "vehicle_id": vehicle_id,
            "steps": steps
        }


def solver(env: LogisticsEnvironment, alpha: float = 0.7, use_2opt: bool = False) -> Dict:
    """
    Main solver function implementing the mathematical model
    
    Args:
        env: LogisticsEnvironment instance
        alpha: Weight for cost vs distance (higher = prioritize cost)
        use_2opt: Whether to apply 2-opt optimization
    
    Returns:
        Solution dictionary with routes
    """
    # Initialize components
    vrp_solver = VRPSolver(env)
    inventory_allocator = InventoryAllocator(env, vrp_solver)
    route_builder = RouteBuilder(env, vrp_solver)
        
    # Phase A: Inventory Allocation
    allocation, fulfilled_orders = inventory_allocator.allocate_inventory()
    fulfillment_rate = len(fulfilled_orders) / len(env.get_all_order_ids()) * 100 if env.get_all_order_ids() else 0
    
    # Phase B: Route Construction
    routes = route_builder.build_routes(allocation, fulfilled_orders)
    
    # Prepare solution
    solution = {"routes": routes}
        
    return solution


if __name__ == "__main__":
    env = LogisticsEnvironment()
    solution = solver(env)
    
    print(f"\n{'='*60}")
    print(f"📊 SOLUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Routes Generated: {len(solution['routes'])}")
    
    # Validate
    is_valid, message = env.validate_solution_business_logic(solution)
    print(f"\n✅ Validation: {message if is_valid else '❌ ' + message}")
    
    # Execute
    success, exec_message = env.execute_solution(solution)
    print(f"{'✅' if success else '❌'} Execution: {exec_message}")
    
    if success:
        # Get statistics
        stats = env.get_solution_fulfillment_summary(solution)
        print(f"\n{'='*60}")
        print(f"📈 PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"💰 Total Cost: £{stats.get('total_cost', 0):,.2f}")
        print(f"📦 Fulfillment Rate: {stats.get('average_fulfillment_rate')}%")
        print(f"✅ Orders Fulfilled: {stats.get('fully_fulfilled_orders', 0)}/{stats.get('total_orders', 0)}")
        print(f"📏 Total Distance: {stats.get('total_distance', 0)} km")
        print(f"🚚 Vehicles Used: {stats.get('vehicles_used', 0)}")
        print(f"{'='*60}")
    else:
        print("\n⚠️  Execution failed - check route details above")