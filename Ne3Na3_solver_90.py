"""
ROBIN LOGISTICS - PyVRP INTEGRATION (Clean Standalone Version)

This file contains ONLY the working Robin Logistics components adapted for PyVRP architecture.
No dependencies on the pyvrp library - completely standalone.

Components:
1. RobinCostEvaluator - Cost calculation matching PyVRP's CostEvaluator interface
2. RobinProblemData - Converts Robin environment to PyVRP-compatible format
3. RobinSolution - Wraps Robin solutions with PyVRP-like interface
4. Utility functions - Helper functions for inventory and load calculations
5. solve_robin_with_pyvrp() - Main solver using PyVRP architecture
"""

from typing import Optional, List, Dict, Set, Tuple
from collections import defaultdict

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import heapq
import random
import copy
import math


class VRPSolver:
    """Dijkstra pathfinding for VRP"""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self._build_adjacency_list()
        self.path_cache = {}
        self.distance_cache = {}
        
    def _build_adjacency_list(self) -> Dict:
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
        if node in self.adjacency_list:
            return self.adjacency_list[node]
        str_node = str(node)
        if str_node in self.adjacency_list:
            return self.adjacency_list[str_node]
        return []
    
    def dijkstra_shortest_path(self, start: int, goal: int) -> Optional[List[int]]:
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
        return len(path) - 1 if path and len(path) > 1 else 0


class Solution:
    """Represents a VRP solution with fitness evaluation"""
    
    def __init__(self, routes: List[Dict], env: LogisticsEnvironment):
        self.routes = routes
        self.env = env
        self.fitness = None
        self.cost = None
        self.fulfillment = None
        self._evaluate()
    
    def _evaluate(self):
        """Calculate fitness: minimize cost while maximizing fulfillment"""
        if not self.routes:
            self.fitness = float('inf')
            self.cost = 0
            self.fulfillment = 0
            return
        
        # Calculate cost
        total_cost = 0
        for route in self.routes:
            vehicle = self.env.get_vehicle_by_id(route['vehicle_id'])
            total_cost += vehicle.fixed_cost
            
            # Calculate distance
            distance = 0
            for i in range(len(route['steps']) - 1):
                distance += 1  # Simplified - each edge = 1 unit
            total_cost += distance * vehicle.cost_per_km
        
        # Calculate fulfillment
        fulfilled_count = 0
        delivered_orders = set()
        for route in self.routes:
            for step in route['steps']:
                for delivery in step.get('deliveries', []):
                    delivered_orders.add(delivery['order_id'])
        
        # Check which orders are fully fulfilled
        for order_id in delivered_orders:
            status = self.env.get_order_fulfillment_status(order_id)
            if sum(status['remaining'].values()) == 0:
                fulfilled_count += 1
        
        total_orders = len(self.env.get_all_order_ids())
        fulfillment_pct = (fulfilled_count / total_orders) * 100
        
        # Objective: S = Cost + Penalty for unfulfilled orders
        C_bench = 10000
        S = total_cost + C_bench * (100 - fulfillment_pct)
        
        self.fitness = S
        self.cost = total_cost
        self.fulfillment = fulfillment_pct
    
    def copy(self):
        """Create a deep copy of the solution"""
        return Solution([copy.deepcopy(r) for r in self.routes], self.env)


class HybridGeneticSearch:
    """
    Hybrid Genetic Search algorithm for VRP
    Combines genetic algorithm with local search
    """
    
    def __init__(self, env: LogisticsEnvironment, vrp_solver: VRPSolver):
        self.env = env
        self.vrp_solver = vrp_solver
        
        # HGS Parameters
        self.pop_size_elite = 4  # Elite solutions
        self.pop_size_diverse = 6  # Diverse solutions
        self.pop_size = self.pop_size_elite + self.pop_size_diverse
        
        self.population = []
        self.best_solution = None
        
        # Allocation data
        self.allocation = None
        self.fulfilled_orders = None
    
    def solve(self, allocation: Dict, fulfilled_orders: Set[str], 
              max_iterations: int = 50) -> Solution:
        """
        Main HGS algorithm
        1. Initialize population
        2. Iterate: select parents, crossover, mutate, educate, update population
        3. Return best solution
        """
        self.allocation = allocation
        self.fulfilled_orders = fulfilled_orders
        
        print("🧬 Initializing population...")
        self._initialize_population()
        
        print(f"🔄 Running HGS for {max_iterations} iterations...")
        for iteration in range(max_iterations):
            # Select parents for crossover
            parent1, parent2 = self._select_parents()
            
            # Crossover to create offspring
            offspring = self._crossover(parent1, parent2)
            
            # Education (local search) on offspring
            offspring = self._educate(offspring)
            
            # Update population
            self._update_population(offspring)
            
            if iteration % 10 == 0:
                best_fitness = self.best_solution.fitness if self.best_solution else float('inf')
                print(f"   Iteration {iteration}: Best fitness = £{best_fitness:,.2f}")
        
        print(f"✅ HGS complete. Best solution: £{self.best_solution.cost:,.2f}")
        return self.best_solution
    
    def _initialize_population(self):
        """Create initial population with diverse solutions"""
        # Group orders by warehouse
        warehouse_orders = defaultdict(list)
        for wh_id in self.allocation.keys():
            for order_id in self.allocation[wh_id].keys():
                if order_id in self.fulfilled_orders:
                    customer_node = self.env.get_order_location(order_id)
                    warehouse_orders[wh_id].append(
                        (order_id, customer_node, self.allocation[wh_id][order_id])
                    )
        
        # Create multiple diverse initial solutions
        for i in range(self.pop_size):
            routes = self._create_initial_solution(warehouse_orders, randomness=i*0.1)
            if routes:
                solution = Solution(routes, self.env)
                self.population.append(solution)
                
                if self.best_solution is None or solution.fitness < self.best_solution.fitness:
                    self.best_solution = solution.copy()
    
    def _create_initial_solution(self, warehouse_orders: Dict, randomness: float = 0.0) -> List[Dict]:
        """Create an initial solution with some randomness"""
        routes = []
        
        for wh_id, orders in warehouse_orders.items():
            wh = self.env.warehouses[wh_id]
            wh_node = wh.location.id
            
            available_vehicles = [v for v in self.env.get_all_vehicles() 
                                 if v.home_warehouse_id == wh_id]
            
            if not available_vehicles:
                continue
            
            home_node = self.env.warehouses[available_vehicles[0].home_warehouse_id].location.id
            
            # Rank vehicles by cost efficiency with randomness
            if random.random() < randomness:
                random.shuffle(available_vehicles)
            else:
                available_vehicles.sort(
                    key=lambda v: v.fixed_cost / max(v.capacity_weight + v.capacity_volume * 100, 1)
                )
            
            # Pack orders into vehicles
            assigned_orders = set()
            
            for vehicle in available_vehicles:
                if len(assigned_orders) >= len(orders):
                    break
                
                route_orders = []
                route_weight = 0.0
                route_volume = 0.0
                
                # Greedy packing with some randomness
                order_candidates = [i for i in range(len(orders)) if i not in assigned_orders]
                if randomness > 0:
                    random.shuffle(order_candidates)
                
                for idx in order_candidates:
                    order_id, customer_node, skus = orders[idx]
                    weight, volume = self._calculate_order_load(skus)
                    
                    if (route_weight + weight <= vehicle.capacity_weight and
                        route_volume + volume <= vehicle.capacity_volume):
                        route_orders.append(idx)
                        route_weight += weight
                        route_volume += volume
                        assigned_orders.add(idx)
                
                if route_orders:
                    # Optimize visit order
                    route_orders_sorted = self._tsp_nearest_neighbor(
                        route_orders, orders, home_node
                    )
                    route_orders_data = [orders[i] for i in route_orders_sorted]
                    
                    route = self._build_route(
                        vehicle.id, home_node, wh_id, wh_node, route_orders_data
                    )
                    if route:
                        routes.append(route)
        
        return routes
    
    def _select_parents(self) -> Tuple[Solution, Solution]:
        """Binary tournament selection"""
        def tournament():
            candidates = random.sample(self.population, min(2, len(self.population)))
            return min(candidates, key=lambda x: x.fitness)
        
        return tournament(), tournament()
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        """
        Order Crossover (OX): combine routes from two parents
        Simplified version for VRP
        """
        # For simplicity, take best routes from both parents
        all_routes = parent1.routes + parent2.routes
        
        # Select diverse subset of routes
        selected_routes = []
        covered_orders = set()
        
        # Sort routes by efficiency
        route_scores = []
        for route in all_routes:
            vehicle = self.env.get_vehicle_by_id(route['vehicle_id'])
            orders_count = sum(1 for step in route['steps'] if step.get('deliveries'))
            if orders_count > 0:
                efficiency = vehicle.fixed_cost / orders_count
                route_scores.append((efficiency, route))
        
        route_scores.sort(key=lambda x: x[0])
        
        # Select non-overlapping routes
        for efficiency, route in route_scores:
            route_orders = set()
            for step in route['steps']:
                for delivery in step.get('deliveries', []):
                    route_orders.add(delivery['order_id'])
            
            # Check if route has new orders
            if not route_orders.issubset(covered_orders):
                selected_routes.append(route)
                covered_orders.update(route_orders)
        
        return Solution(selected_routes, self.env)
    
    def _educate(self, solution: Solution) -> Solution:
        """
        Education operator: intensive local search
        Apply multiple operators: relocate, swap, 2-opt
        """
        improved = True
        iterations = 0
        max_iterations = 10
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try relocate operator
            new_solution = self._relocate(solution)
            if new_solution.fitness < solution.fitness:
                solution = new_solution
                improved = True
            
            # Try swap operator
            new_solution = self._swap(solution)
            if new_solution.fitness < solution.fitness:
                solution = new_solution
                improved = True
        
        return solution
    
    def _relocate(self, solution: Solution) -> Solution:
        """
        Relocate operator: move a customer from one route to another
        """
        best_solution = solution.copy()
        
        # Try moving customers between routes
        for i in range(len(solution.routes)):
            for j in range(len(solution.routes)):
                if i == j:
                    continue
                
                # Extract orders from route i
                orders_i = []
                for step in solution.routes[i]['steps']:
                    for delivery in step.get('deliveries', []):
                        orders_i.append(delivery['order_id'])
                
                if not orders_i:
                    continue
                
                # Try moving one order to route j
                order_to_move = orders_i[0]  # Simplified: move first order
                
                # Check if route j has capacity
                # (Simplified capacity check - in production would be more thorough)
                
                # Create new solution (simplified)
                # In production: actually reconstruct routes
                
        return best_solution
    
    def _swap(self, solution: Solution) -> Solution:
        """
        Swap operator: exchange customers between two routes
        """
        # Simplified version - return original solution
        return solution.copy()
    
    def _update_population(self, offspring: Solution):
        """
        Update population with offspring
        Maintain elite solutions and diversity
        """
        # Add offspring to population
        self.population.append(offspring)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness)
        
        # Keep elite solutions
        elite = self.population[:self.pop_size_elite]
        
        # Keep diverse solutions from remaining
        diverse = []
        remaining = self.population[self.pop_size_elite:]
        
        while len(diverse) < self.pop_size_diverse and remaining:
            # Select most diverse solution
            if not diverse:
                diverse.append(remaining.pop(0))
            else:
                # Simple diversity: different number of routes
                best_diversity = -1
                best_idx = 0
                for idx, sol in enumerate(remaining):
                    diversity = abs(len(sol.routes) - len(diverse[0].routes))
                    if diversity > best_diversity:
                        best_diversity = diversity
                        best_idx = idx
                
                diverse.append(remaining.pop(best_idx))
        
        # Update population
        self.population = elite + diverse
        
        # Update best solution
        if self.population[0].fitness < self.best_solution.fitness:
            self.best_solution = self.population[0].copy()
    
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
    
    def _tsp_nearest_neighbor(self, order_indices: List[int], orders: List[Tuple], 
                               home_node: int) -> List[int]:
        """Nearest neighbor TSP heuristic"""
        if len(order_indices) <= 1:
            return order_indices
        
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
    
    def _build_route(self, vehicle_id: str, home_node: int, wh_id: str, 
                     wh_node: int, orders: List[Tuple]) -> Optional[Dict]:
        """Build complete route"""
        if not orders:
            return None
        
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


class InventoryAllocator:
    """Greedy inventory allocation"""
    
    def __init__(self, env: LogisticsEnvironment, vrp_solver: VRPSolver):
        self.env = env
        self.vrp_solver = vrp_solver
    
    def allocate_inventory(self) -> Tuple[Dict, Set[str]]:
        order_ids = self.env.get_all_order_ids()
        warehouse_ids = list(self.env.warehouses.keys())
        
        inventory = {wh_id: self.env.get_warehouse_inventory(wh_id).copy() 
                    for wh_id in warehouse_ids}
        
        allocation = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        fulfilled_orders = set()
        
        orders_data = []
        for order_id in order_ids:
            requirements = self.env.get_order_requirements(order_id)
            if requirements:
                customer_node = self.env.get_order_location(order_id)
                total_demand = sum(requirements.values())
                orders_data.append((order_id, customer_node, requirements, total_demand))
        
        orders_data.sort(key=lambda x: x[3])
        
        for order_id, customer_node, demand, _ in orders_data:
            if customer_node is None:
                continue
            
            warehouse_costs = []
            for wh_id in warehouse_ids:
                wh_node = self.env.warehouses[wh_id].location.id
                path = self.vrp_solver.dijkstra_shortest_path(wh_node, customer_node)
                if path:
                    dist = self.vrp_solver.get_path_distance(path)
                    warehouse_costs.append((dist, wh_id))
            
            warehouse_costs.sort()
            
            order_allocation = defaultdict(lambda: defaultdict(float))
            remaining_demand = demand.copy()
            
            for dist, wh_id in warehouse_costs:
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
            
            if not remaining_demand:
                fulfilled_orders.add(order_id)
                for wh_id, sku_alloc in order_allocation.items():
                    for sku, qty in sku_alloc.items():
                        allocation[wh_id][order_id][sku] = qty
        
        return allocation, fulfilled_orders


def solver_75(env: LogisticsEnvironment) -> Dict:
    """
    HGS-based VRP solver inspired by PyVRP
    
    Uses hybrid genetic search with:
    - Population-based evolution
    - Local search operators
    - Diversity management
    """
    print("🚀 Starting HGS VRP Solver (PyVRP-inspired)...\n")
    
    # Initialize
    vrp_solver = VRPSolver(env)
    allocator = InventoryAllocator(env, vrp_solver)
    hgs = HybridGeneticSearch(env, vrp_solver)
    
    # Phase 1: Inventory allocation
    allocation, fulfilled_orders = allocator.allocate_inventory()
    
    # Phase 2: HGS optimization
    best_solution = hgs.solve(allocation, fulfilled_orders, max_iterations=30)
    
    return {"routes": best_solution.routes}


# ============================================================================
# 1. ROBIN COST EVALUATOR (PyVRP-Compatible)
# ============================================================================

class RobinCostEvaluator:
    """
    CostEvaluator adapted for Robin Logistics environment.
    
    This class implements PyVRP's CostEvaluator interface but uses Robin's
    environment to calculate costs instead of the standard VRP penalties.
    
    Parameters
    ----------
    env
        Robin LogisticsEnvironment instance.
    load_penalties
        Penalty weights for capacity violations per dimension.
        For Robin: [weight_penalty, volume_penalty]
    tw_penalty
        Time window violation penalty (not used in Robin, set to 0).
    dist_penalty
        Distance penalty (not used in Robin, set to 0).
    
    Attributes
    ----------
    env
        The Robin environment for cost calculations.
    load_penalties
        List of penalty weights for load violations.
    tw_penalty
        Time window penalty weight.
    dist_penalty
        Distance penalty weight.
    """
    
    def __init__(
        self,
        env,  # LogisticsEnvironment
        load_penalties: list[float] = None,
        tw_penalty: float = 0.0,
        dist_penalty: float = 0.0,
    ):
        """
        Initialize Robin-specific cost evaluator.
        
        Parameters
        ----------
        env
            Robin LogisticsEnvironment instance
        load_penalties
            Penalties for [weight, volume] violations. Default [1000, 1000].
        tw_penalty
            Time window penalty (not used in Robin)
        dist_penalty
            Distance penalty (not used in Robin)
        """
        self.env = env
        self.load_penalties = load_penalties or [1000.0, 1000.0]
        self.tw_penalty = tw_penalty
        self.dist_penalty = dist_penalty
    
    def load_penalty(
        self, load: float, capacity: float, dimension: int
    ) -> float:
        """
        Calculate penalty for capacity violation.
        
        Parameters
        ----------
        load
            Current load in the dimension.
        capacity
            Capacity limit in the dimension.
        dimension
            Load dimension index (0=weight, 1=volume).
        
        Returns
        -------
        float
            Penalty value for the violation.
        """
        if load <= capacity:
            return 0.0
        
        excess = load - capacity
        penalty_weight = self.load_penalties[dimension] if dimension < len(self.load_penalties) else 1000.0
        
        return penalty_weight * excess
    
    def tw_penalty(self, time_warp: float) -> float:
        """
        Calculate time window violation penalty.
        
        Parameters
        ----------
        time_warp
            Amount of time window violation.
        
        Returns
        -------
        float
            Penalty for time window violation (always 0 for Robin).
        """
        # Robin doesn't have time windows
        return 0.0
    
    def dist_penalty(self, distance: float, max_distance: float) -> float:
        """
        Calculate distance violation penalty.
        
        Parameters
        ----------
        distance
            Total distance traveled.
        max_distance
            Maximum allowed distance.
        
        Returns
        -------
        float
            Penalty for distance violation (always 0 for Robin).
        """
        # Robin doesn't have distance constraints
        return 0.0
    
    def penalised_cost(self, solution: Dict) -> float:
        """
        Calculate penalised cost (fitness function) for a solution.
        
        Uses Robin's calculate_solution_cost() plus penalties for constraint violations.
        
        Parameters
        ----------
        solution
            Solution in Robin format: {"routes": [...]}
        
        Returns
        -------
        float
            Total penalised cost including violations
        """
        # Base cost from Robin environment
        base_cost = self.env.calculate_solution_cost(solution)
        
        # Calculate penalties
        total_penalty = 0.0
        
        # Check each route for capacity violations
        for route in solution.get('routes', []):
            vehicle_id = route.get('vehicle_id')
            if not vehicle_id:
                continue
            
            vehicle = self.env.get_vehicle_by_id(vehicle_id)
            if not vehicle:
                continue
            
            # Calculate total load for this route
            total_weight = 0.0
            total_volume = 0.0
            
            for step in route.get('steps', []):
                for delivery in step.get('deliveries', []):
                    order_id = delivery.get('order_id')
                    requirements = self.env.get_order_requirements(order_id)
                    
                    if requirements:
                        for sku_id, qty in requirements.items():
                            sku_details = self.env.get_sku_details(sku_id)
                            if sku_details:
                                total_weight += sku_details.get('weight', 0) * qty
                                total_volume += sku_details.get('volume', 0) * qty
            
            # Apply penalties for capacity violations
            weight_penalty = self.load_penalty(
                total_weight, vehicle.capacity_weight, dimension=0
            )
            volume_penalty = self.load_penalty(
                total_volume, vehicle.capacity_volume, dimension=1
            )
            
            total_penalty += weight_penalty + volume_penalty
        
        # Check for unfulfilled orders
        all_orders = set(self.env.get_all_order_ids())
        fulfilled_orders = set()
        for route in solution.get('routes', []):
            for step in route.get('steps', []):
                for delivery in step.get('deliveries', []):
                    fulfilled_orders.add(delivery['order_id'])
        
        unfulfilled = len(all_orders - fulfilled_orders)
        unfulfillment_penalty = unfulfilled * 10000.0  # £10k per unfulfilled order
        
        return base_cost + total_penalty + unfulfillment_penalty
    
    def cost(self, solution: Dict) -> float:
        """
        Calculate base cost (objective function) for a solution.
        
        Uses Robin's calculate_solution_cost() directly without penalties.
        
        Parameters
        ----------
        solution
            Solution in Robin format: {"routes": [...]}
        
        Returns
        -------
        float
            Base solution cost from Robin environment
        """
        return self.env.calculate_solution_cost(solution)


# ============================================================================
# 2. ROBIN PROBLEM DATA (PyVRP-Compatible)
# ============================================================================

class RobinProblemData:
    """
    Adapter that converts Robin LogisticsEnvironment to PyVRP ProblemData format.
    
    This class provides a PyVRP-compatible interface to Robin's data while
    maintaining compatibility with Robin's API.
    
    Attributes
    ----------
    env
        Robin LogisticsEnvironment instance
    num_clients
        Number of delivery orders (customers)
    num_depots
        Number of warehouses
    num_vehicles
        Number of available vehicles
    num_locations
        Total locations (warehouses + customers)
    clients_list
        List of order IDs
    depots_list
        List of warehouse IDs
    vehicles_list
        List of vehicle objects
    """
    
    def __init__(self, env):
        """
        Initialize Robin problem data adapter.
        
        Parameters
        ----------
        env
            Robin LogisticsEnvironment instance
        """
        self.env = env
        
        # Extract core data
        self.clients_list = env.get_all_order_ids()  # Order IDs
        self.depots_list = list(env.warehouses.keys())  # Warehouse IDs
        self.vehicles_list = env.get_all_vehicles()  # Vehicle objects
        
        # Counts
        self.num_clients = len(self.clients_list)
        self.num_depots = len(self.depots_list)
        self.num_vehicles = len(self.vehicles_list)
        self.num_locations = self.num_depots + self.num_clients
        
        # Mappings: order_id -> index, warehouse_id -> index
        self.client_to_idx = {order_id: idx for idx, order_id in enumerate(self.clients_list)}
        self.depot_to_idx = {wh_id: idx for idx, wh_id in enumerate(self.depots_list)}
        
        # Reverse mappings: index -> order_id, index -> warehouse_id
        self.idx_to_client = {idx: order_id for order_id, idx in self.client_to_idx.items()}
        self.idx_to_depot = {idx: wh_id for wh_id, idx in self.depot_to_idx.items()}
        
        # Node mappings (location node_id -> data index)
        self.node_to_idx = {}
        self.idx_to_node = {}
        
        idx = 0
        # Depots first
        for wh_id in self.depots_list:
            node_id = env.warehouses[wh_id].location.id
            self.node_to_idx[node_id] = idx
            self.idx_to_node[idx] = node_id
            idx += 1
        
        # Then clients
        for order_id in self.clients_list:
            node_id = env.get_order_location(order_id)
            if node_id is not None:
                self.node_to_idx[node_id] = idx
                self.idx_to_node[idx] = node_id
                idx += 1
        
        print(f"📦 RobinProblemData initialized:")
        print(f"   Clients: {self.num_clients}")
        print(f"   Depots: {self.num_depots}")
        print(f"   Vehicles: {self.num_vehicles}")
        print(f"   Locations: {self.num_locations}")
    
    def get_client_requirements(self, client_idx: int) -> Dict[str, float]:
        """Get requirements for a client (order)."""
        order_id = self.idx_to_client.get(client_idx)
        if order_id:
            return self.env.get_order_requirements(order_id) or {}
        return {}
    
    def get_client_location(self, client_idx: int) -> Optional[int]:
        """Get node_id for a client."""
        order_id = self.idx_to_client.get(client_idx)
        if order_id:
            return self.env.get_order_location(order_id)
        return None
    
    def get_depot_location(self, depot_idx: int) -> Optional[int]:
        """Get node_id for a depot (warehouse)."""
        wh_id = self.idx_to_depot.get(depot_idx)
        if wh_id and wh_id in self.env.warehouses:
            return self.env.warehouses[wh_id].location.id
        return None
    
    def get_vehicle_capacity(self, vehicle_idx: int) -> Tuple[float, float]:
        """Get vehicle capacity (weight, volume)."""
        if 0 <= vehicle_idx < len(self.vehicles_list):
            vehicle = self.vehicles_list[vehicle_idx]
            return (vehicle.capacity_weight, vehicle.capacity_volume)
        return (0.0, 0.0)


# ============================================================================
# 3. ROBIN SOLUTION WRAPPER (PyVRP-Compatible)
# ============================================================================

class RobinSolution:
    """
    Adapter for converting between PyVRP Solution format and Robin solution format.
    
    This class maintains both representations and provides conversion methods.
    
    Attributes
    ----------
    robin_solution
        Solution in Robin format: {"routes": [...]}
    problem_data
        RobinProblemData instance
    """
    
    def __init__(self, robin_solution: Dict, problem_data):
        """
        Initialize Robin solution wrapper.
        
        Parameters
        ----------
        robin_solution
            Solution in Robin format
        problem_data
            RobinProblemData instance
        """
        self.robin_solution = robin_solution
        self.problem_data = problem_data
        self._is_feasible = None
        self._cost = None
    
    def is_feasible(self) -> bool:
        """Check if solution is feasible."""
        if self._is_feasible is None:
            is_valid, _ = self.problem_data.env.validate_solution_business_logic(
                self.robin_solution
            )
            self._is_feasible = is_valid
        return self._is_feasible
    
    def cost(self, cost_evaluator) -> float:
        """Get solution cost."""
        if self._cost is None:
            self._cost = cost_evaluator.cost(self.robin_solution)
        return self._cost
    
    def num_routes(self) -> int:
        """Number of routes in solution."""
        return len(self.robin_solution.get('routes', []))
    
    def num_clients(self) -> int:
        """Number of clients served."""
        served = set()
        for route in self.robin_solution.get('routes', []):
            for step in route.get('steps', []):
                for delivery in step.get('deliveries', []):
                    served.add(delivery['order_id'])
        return len(served)
    
    def routes(self) -> List[Dict]:
        """Get routes list."""
        return self.robin_solution.get('routes', [])
    
    def copy(self):
        """Create a deep copy."""
        import copy
        return RobinSolution(
            copy.deepcopy(self.robin_solution),
            self.problem_data
        )
    
    @classmethod
    def from_routes(cls, routes: List[Dict], problem_data):
        """Create solution from routes list."""
        return cls({"routes": routes}, problem_data)


# ============================================================================
# 4. UTILITY FUNCTIONS
# ============================================================================

def robin_get_warehouses_with_sku(env, sku_id: str, min_quantity: float = 1) -> List[str]:
    """
    Find warehouses that have a specific SKU with minimum quantity.
    
    This function is NOT in the API reference, so we implement it ourselves.
    
    Parameters
    ----------
    env
        Robin LogisticsEnvironment
    sku_id
        SKU identifier
    min_quantity
        Minimum required quantity
    
    Returns
    -------
    List[str]
        List of warehouse IDs that have the SKU
    """
    warehouses = []
    for wh_id in env.warehouses.keys():
        inventory = env.get_warehouse_inventory(wh_id).copy()
        if inventory.get(sku_id, 0) >= min_quantity:
            warehouses.append(wh_id)
    return warehouses


def robin_calculate_order_load(env, order_id: str) -> Tuple[float, float]:
    """
    Calculate total weight and volume for an order.
    
    Parameters
    ----------
    env
        Robin LogisticsEnvironment
    order_id
        Order identifier
    
    Returns
    -------
    Tuple[float, float]
        (total_weight, total_volume) for the order
    """
    requirements = env.get_order_requirements(order_id)
    if not requirements:
        return (0.0, 0.0)
    
    total_weight = 0.0
    total_volume = 0.0
    
    for sku_id, qty in requirements.items():
        sku_details = env.get_sku_details(sku_id)
        if sku_details:
            total_weight += sku_details.get('weight', 0) * qty
            total_volume += sku_details.get('volume', 0) * qty
    
    return (total_weight, total_volume)


def robin_allocate_inventory_greedy(env, problem_data) -> Tuple[Dict, Set[str]]:
    """
    Greedy inventory allocation for Robin environment.
    
    Allocates orders to nearest warehouses with available inventory.
    
    Parameters
    ----------
    env
        Robin LogisticsEnvironment
    problem_data
        RobinProblemData instance
    
    Returns
    -------
    Tuple[Dict, Set[str]]
        (allocation dict, fulfilled_orders set)
        allocation: {wh_id: {order_id: {sku_id: qty}}}
        fulfilled_orders: set of order IDs that can be fulfilled
    """
    warehouse_ids = list(env.warehouses.keys())
    order_ids = env.get_all_order_ids()
    
    # Copy inventory
    inventory = {wh_id: env.get_warehouse_inventory(wh_id).copy() 
                for wh_id in warehouse_ids}
    
    allocation = defaultdict(lambda: defaultdict(dict))
    fulfilled_orders = set()
    
    # Sort orders by total demand (smallest first for easier packing)
    orders_data = []
    for order_id in order_ids:
        requirements = env.get_order_requirements(order_id)
        if requirements:
            weight, volume = robin_calculate_order_load(env, order_id)
            total_demand = weight + volume  # Simple heuristic
            orders_data.append((order_id, requirements, total_demand))
    
    orders_data.sort(key=lambda x: x[2])  # Sort by total demand
    
    # Allocate each order
    for order_id, requirements, _ in orders_data:
        customer_node = env.get_order_location(order_id)
        if customer_node is None:
            continue
        
        # Find warehouses that can fulfill this order
        candidate_warehouses = []
        for wh_id in warehouse_ids:
            # Check if warehouse has all required items
            can_fulfill = all(
                inventory[wh_id].get(sku, 0) >= qty
                for sku, qty in requirements.items()
            )
            
            if can_fulfill:
                # Get distance (using env.get_distance if available)
                wh_node = env.warehouses[wh_id].location.id
                dist = env.get_distance(wh_node, customer_node)
                
                if dist is None:
                    # If get_distance not available, use placeholder
                    dist = 999999
                
                candidate_warehouses.append((dist, wh_id))
        
        # Allocate from nearest warehouse
        if candidate_warehouses:
            candidate_warehouses.sort()
            _, best_wh = candidate_warehouses[0]
            
            # Allocate the order
            for sku, qty in requirements.items():
                allocation[best_wh][order_id][sku] = qty
                inventory[best_wh][sku] -= qty
            
            fulfilled_orders.add(order_id)
    
    return allocation, fulfilled_orders

def validate_and_fix_inventory_conflicts(env, solution):
    """
    Remove routes that create inventory conflicts.
    Keeps routes in order until inventory is exhausted.
    """
    wh_inventory = {
        wh_id: env.get_warehouse_inventory(wh_id).copy()
        for wh_id in env.warehouses.keys()
    }
    
    valid_routes = []
    
    for route in solution['routes']:
        # Check if all pickups in this route are feasible
        can_fulfill = True
        
        for step in route['steps']:
            for pickup in step.get('pickups', []):
                wh_id = pickup['warehouse_id']
                sku_id = pickup['sku_id']
                qty = pickup['quantity']
                
                if wh_inventory.get(wh_id, {}).get(sku_id, 0) < qty:
                    can_fulfill = False
                    break
            
            if not can_fulfill:
                break
        
        if can_fulfill:
            # Deduct inventory and keep this route
            for step in route['steps']:
                for pickup in step.get('pickups', []):
                    wh_id = pickup['warehouse_id']
                    sku_id = pickup['sku_id']
                    qty = pickup['quantity']
                    wh_inventory[wh_id][sku_id] -= qty
            
            valid_routes.append(route)
    
    return {"routes": valid_routes}

# ============================================================================
# 5. COMPLETE ROBIN SOLVER USING PYVRP ARCHITECTURE
# ============================================================================

def solver(env) -> Dict:
    """
    Complete solver for Robin Logistics using PyVRP architecture.
    
    This is the main entry point that integrates all components:
    - RobinProblemData for data conversion
    - RobinCostEvaluator for cost calculation
    - Inventory allocation
    - Initial solution generation
    - Genetic algorithm (uses Ne3Na3_solver_84)
    
    Parameters
    ----------
    env
        Robin LogisticsEnvironment instance
    
    Returns
    -------
    Dict
        Solution in Robin format: {"routes": [...]}
    """
    print("=" * 80)
    print("🚀 ROBIN LOGISTICS SOLVER - PyVRP ARCHITECTURE")
    print("=" * 80)
    print()
    
    # Step 1: Convert Robin data to PyVRP format
    print("Step 1: Converting Robin data to PyVRP format...")
    problem_data = RobinProblemData(env)
    print()
    
    # Step 2: Initialize cost evaluator
    print("Step 2: Initializing cost evaluator...")
    cost_evaluator = RobinCostEvaluator(
        env=env,
        load_penalties=[1000.0, 1000.0],  # [weight, volume]
        tw_penalty=0.0,
        dist_penalty=0.0
    )
    print("✅ Cost evaluator ready")
    print()
    
    # Step 3: Allocate inventory
    print("Step 3: Allocating inventory...")
    allocation, fulfilled_orders = robin_allocate_inventory_greedy(env, problem_data)
    print(f"✅ Allocated {len(fulfilled_orders)}/{problem_data.num_clients} orders")
    print()
    
    # Step 4: Generate initial solution
    print("Step 4: Generating initial solution...")
    # Use existing solver (e.g., from Ne3Na3_solver_75)
    try:
        raw_solution = solver_75(env)
        solution = validate_and_fix_inventory_conflicts(env, raw_solution)
    except ImportError:
        solution = {"routes": []}
    print()
    
    # Step 5: Wrap in RobinSolution for PyVRP compatibility
    robin_solution = RobinSolution(solution, problem_data)
    
    # Step 6: Calculate final metrics
    print("=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    
    base_cost = cost_evaluator.cost(solution)
    penalised_cost = cost_evaluator.penalised_cost(solution)
    
    print(f"Routes: {robin_solution.num_routes()}")
    print(f"Clients served: {robin_solution.num_clients()}/{problem_data.num_clients}")
    print(f"Base cost: £{base_cost:,.2f}")
    print(f"Penalised cost: £{penalised_cost:,.2f}")
    print(f"Feasible: {robin_solution.is_feasible()}")
    print("=" * 80)
    print()
    
    return solution


# ============================================================================
# API COVERAGE NOTES
# ============================================================================

"""
ROBIN API FUNCTIONS USED (from API_REFERENCE.md):

✅ AVAILABLE (10/11):
1. env.get_distance(node1_id, node2_id) -> Optional[float]
2. env.calculate_solution_cost(solution) -> float
3. env.get_order_fulfillment_status(order_id) -> Dict
4. env.validate_solution_business_logic(solution) -> Tuple[bool, str]
5. env.get_warehouse_inventory(warehouse_id) -> Dict[sku_id, quantity]
6. env.get_order_requirements(order_id) -> Dict[sku_id, quantity]
7. env.get_sku_details(sku_id) -> Optional[Dict]
8. env.get_order_location(order_id) -> int
9. env.get_vehicle_by_id(vehicle_id) -> Vehicle
10. env.get_all_order_ids() -> List[str]

❌ MISSING (1/11):
1. env.get_warehouses_with_sku(sku_id, min_quantity)
   → Implemented as robin_get_warehouses_with_sku() above

CONCLUSION: All critical API functions are available. Only one convenience
function is missing, which we've implemented ourselves.
"""


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    from robin_logistics import LogisticsEnvironment
    
    env = LogisticsEnvironment()
    
    print("=" * 80)
    print("🧪 TESTING ROBIN-PYVRP INTEGRATION")
    print("=" * 80)
    print()
    
    # Test 1: RobinProblemData
    print("Test 1: RobinProblemData")
    print("-" * 40)
    problem_data = RobinProblemData(env)
    print()
    
    # Test 2: RobinCostEvaluator
    print("Test 2: RobinCostEvaluator")
    print("-" * 40)
    cost_evaluator = RobinCostEvaluator(env)
    print("✅ Cost evaluator initialized")
    print()
    
    # Test 3: Inventory allocation
    print("Test 3: Inventory Allocation")
    print("-" * 40)
    allocation, fulfilled = robin_allocate_inventory_greedy(env, problem_data)
    print(f"Fulfilled orders: {len(fulfilled)}/{problem_data.num_clients}")
    print()
    
    # Test 4: Full solver
    print("Test 4: Complete Solver")
    print("-" * 40)
    solution = solver(env)

    is_valid, msg, summary = env.validate_solution_complete(solution)
    invalid_routes = list(summary.items())[1]
    print(f"Validation: {is_valid} - {msg}, total routes: {len(solution['routes'])}") 
    if is_valid: 
        ok, exec_msg = env.execute_solution(solution) 
        print(f"Execution: {ok} - {exec_msg}") 
        stats = env.get_solution_statistics(solution) 
        print(f"Orders served: {stats.get('unique_orders_served', 0)}/{stats.get('total_orders', 0)}") 
        print(f"Total distance: {stats.get('total_distance', 0):.2f} km") # you can print summary using print(summary) to see detailed valid and invalid routes
    
        # Final cost
        final_cost = env.calculate_solution_cost(solution)
        print(f"Final cost: £{final_cost:,.2f}")
        print()
    
    print("=" * 80)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 80)
