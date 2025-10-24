#!/usr/bin/env python3
"""
PyVRP-Inspired Solver for Robin Logistics Environment
Implements core PyVRP concepts adapted for the competition framework

Architecture:
1. ProblemData - Converts Robin env to PyVRP-compatible format
2. CostEvaluator - Evaluates solutions with penalties
3. Solution - Represents routes with feasibility tracking
4. PenaltyManager - Adaptive penalty management
5. Population - Maintains diverse solution pool
6. GeneticAlgorithm - HGS with SREX crossover
7. LocalSearch - Efficient neighborhood search
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
import heapq
import random
import copy
import math
import time


# ============================================================================
# PART 1: PROBLEM DATA CONVERSION
# ============================================================================

class ProblemData:
    """Converts Robin LogisticsEnvironment to PyVRP-compatible problem data"""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.num_clients = len(env.get_all_order_ids())
        self.num_depots = len(env.warehouses)
        self.num_vehicles = len(env.get_all_vehicles())
        self.num_locations = self.num_depots + self.num_clients
        
        # Build mappings
        self.order_ids = env.get_all_order_ids()
        self.order_to_idx = {oid: i for i, oid in enumerate(self.order_ids)}
        self.idx_to_order = {i: oid for oid, i in self.order_to_idx.items()}
        
        # Build distance/duration matrices (simplified - use road network)
        self._build_matrices()
    
    def _build_matrices(self):
        """Build distance and duration matrices"""
        # Simplified: use uniform costs for now
        # In production, would compute actual shortest paths
        self.distance_matrix = [[1.0 for _ in range(self.num_locations)] 
                               for _ in range(self.num_locations)]
        self.duration_matrix = [[1.0 for _ in range(self.num_locations)] 
                               for _ in range(self.num_locations)]


# ============================================================================
# PART 2: COST EVALUATION
# ============================================================================

@dataclass
class CostEvaluator:
    """Evaluates solution costs with capacity and time window penalties"""
    load_penalty: float = 1000.0
    tw_penalty: float = 1000.0
    dist_penalty: float = 1.0
    
    def cost(self, solution: 'Solution') -> float:
        """Calculate total penalized cost"""
        if not solution.routes:
            return float('inf')
        
        # Use Robin's built-in cost calculator for accuracy
        try:
            robin_solution = {"routes": solution.routes}
            actual_cost = solution.env.calculate_solution_cost(robin_solution)
            
            if actual_cost is not None and actual_cost > 0:
                base_cost = actual_cost
            else:
                # Fallback to manual calculation
                base_cost = self._manual_cost_calculation(solution)
        except:
            base_cost = self._manual_cost_calculation(solution)
        
        # Add penalties
        penalty_cost = 0.0
        
        # Capacity penalties
        for route in solution.routes:
            vehicle = solution.env.get_vehicle_by_id(route['vehicle_id'])
            route_weight = 0.0
            route_volume = 0.0
            
            for step in route['steps']:
                for delivery in step.get('deliveries', []):
                    sku = solution.env.skus.get(delivery['sku_id'])
                    if sku:
                        route_weight += sku.weight * delivery['quantity']
                        route_volume += sku.volume * delivery['quantity']
            
            # Add penalties for excess
            if route_weight > vehicle.capacity_weight:
                penalty_cost += self.load_penalty * (route_weight - vehicle.capacity_weight)
            if route_volume > vehicle.capacity_volume:
                penalty_cost += self.load_penalty * (route_volume - vehicle.capacity_volume)
        
        # Fulfillment penalty - check orders in ROUTES not environment state
        # (solver doesn't execute routes, just returns them)
        orders_in_routes = set()
        for route in solution.routes:
            for step in route['steps']:
                for delivery in step.get('deliveries', []):
                    orders_in_routes.add(delivery['order_id'])
        
        total_orders = len(solution.env.get_all_order_ids())
        fulfilled_in_routes = len(orders_in_routes)
        unfulfilled = total_orders - fulfilled_in_routes
        penalty_cost += unfulfilled * 10000  # High penalty for unfulfilled
        
        return base_cost + penalty_cost
    
    def _manual_cost_calculation(self, solution: 'Solution') -> float:
        """Manual cost calculation as fallback"""
        fixed_cost = 0.0
        variable_cost = 0.0
        
        for route in solution.routes:
            vehicle = solution.env.get_vehicle_by_id(route['vehicle_id'])
            fixed_cost += vehicle.fixed_cost
            
            # Calculate distance (number of steps - 1)
            distance = max(0, len(route['steps']) - 1)
            variable_cost += distance * vehicle.cost_per_km
        
        return fixed_cost + variable_cost


# ============================================================================
# PART 3: SOLUTION REPRESENTATION
# ============================================================================

class Solution:
    """Represents a VRP solution with routes"""
    
    def __init__(self, routes: List[Dict], env: LogisticsEnvironment):
        self.routes = routes
        self.env = env
        self._cost = None
        self._is_feasible = None
    
    def copy(self) -> 'Solution':
        """Deep copy of solution"""
        return Solution([copy.deepcopy(r) for r in self.routes], self.env)
    
    def is_feasible(self) -> bool:
        """Check if solution is feasible"""
        if self._is_feasible is not None:
            return self._is_feasible
        
        # Check capacity constraints
        for route in self.routes:
            vehicle = self.env.get_vehicle_by_id(route['vehicle_id'])
            route_weight = 0.0
            route_volume = 0.0
            
            for step in route['steps']:
                for delivery in step.get('deliveries', []):
                    sku = self.env.get_sku_details(delivery['sku_id'])
                    if sku:
                        route_weight += sku.get('weight', 0) * delivery['quantity']
                        route_volume += sku.get('volume', 0) * delivery['quantity']
            
            if (route_weight > vehicle.capacity_weight or 
                route_volume > vehicle.capacity_volume):
                self._is_feasible = False
                return False
        
        # Check fulfillment
        fulfilled = 0
        for order_id in self.env.get_all_order_ids():
            status = self.env.get_order_fulfillment_status(order_id)
            if sum(status['remaining'].values()) == 0:
                fulfilled += 1
        
        total = len(self.env.get_all_order_ids())
        self._is_feasible = (fulfilled == total)
        return self._is_feasible


# ============================================================================
# PART 4: PENALTY MANAGER (Adaptive Penalties)
# ============================================================================

@dataclass
class PenaltyParams:
    """Penalty manager parameters"""
    repair_booster: int = 12
    solutions_between_updates: int = 50
    penalty_increase: float = 1.34
    penalty_decrease: float = 0.32
    target_feasible: float = 0.43
    min_penalty: float = 0.1
    max_penalty: float = 100_000.0


class PenaltyManager:
    """Manages adaptive penalties for constraints"""
    
    def __init__(self, params: PenaltyParams = PenaltyParams()):
        self.params = params
        self.load_penalty = 1000.0
        self.tw_penalty = 1000.0
        self.dist_penalty = 1.0
        self.history: List[bool] = []
    
    def register(self, solution: Solution):
        """Register solution feasibility"""
        self.history.append(solution.is_feasible())
        
        if len(self.history) >= self.params.solutions_between_updates:
            feas_rate = sum(self.history) / len(self.history)
            self._update_penalties(feas_rate)
            self.history.clear()
    
    def _update_penalties(self, feas_rate: float):
        """Update penalties based on feasibility rate"""
        if feas_rate < self.params.target_feasible:
            # Too many infeasible - increase penalties
            self.load_penalty = min(
                self.load_penalty * self.params.penalty_increase,
                self.params.max_penalty
            )
        elif feas_rate > self.params.target_feasible + 0.1:
            # Too many feasible - decrease penalties
            self.load_penalty = max(
                self.load_penalty * self.params.penalty_decrease,
                self.params.min_penalty
            )
    
    def cost_evaluator(self) -> CostEvaluator:
        """Get current cost evaluator"""
        return CostEvaluator(self.load_penalty, self.tw_penalty, self.dist_penalty)
    
    def booster_cost_evaluator(self) -> CostEvaluator:
        """Get boosted cost evaluator for repair"""
        return CostEvaluator(
            self.load_penalty * self.params.repair_booster,
            self.tw_penalty * self.params.repair_booster,
            self.dist_penalty
        )


# ============================================================================
# PART 5: POPULATION MANAGEMENT
# ============================================================================

@dataclass
class PopulationParams:
    """Population parameters"""
    min_pop_size: int = 25
    generation_size: int = 40
    num_elite: int = 4
    lb_diversity: float = 0.1
    ub_diversity: float = 0.5


class Population:
    """Maintains diverse population of solutions"""
    
    def __init__(self, params: PopulationParams = PopulationParams()):
        self.params = params
        self.solutions: List[Tuple[Solution, float]] = []  # (solution, fitness)
    
    def add(self, solution: Solution, cost_eval: CostEvaluator):
        """Add solution to population"""
        cost = cost_eval.cost(solution)
        self.solutions.append((solution.copy(), cost))
        self.solutions.sort(key=lambda x: x[1])
        
        # Keep population size bounded
        if len(self.solutions) > self.params.min_pop_size + self.params.generation_size:
            self.solutions = self.solutions[:self.params.min_pop_size]
    
    def select(self) -> Tuple[Solution, Solution]:
        """Binary tournament selection"""
        if len(self.solutions) < 2:
            # Return copies of first solution
            sol = self.solutions[0][0] if self.solutions else None
            return (sol.copy() if sol else None, sol.copy() if sol else None)
        
        # Tournament selection
        idx1, idx2 = random.sample(range(len(self.solutions)), 2)
        parent1 = self.solutions[idx1][0] if self.solutions[idx1][1] < self.solutions[idx2][1] else self.solutions[idx2][0]
        
        idx3, idx4 = random.sample(range(len(self.solutions)), 2)
        parent2 = self.solutions[idx3][0] if self.solutions[idx3][1] < self.solutions[idx4][1] else self.solutions[idx4][0]
        
        return (parent1.copy(), parent2.copy())
    
    def clear(self):
        """Clear population"""
        self.solutions.clear()
    
    def best(self) -> Optional[Solution]:
        """Get best solution"""
        return self.solutions[0][0].copy() if self.solutions else None


# ============================================================================
# PART 6: CROSSOVER OPERATORS
# ============================================================================

def selective_route_exchange(
    parents: Tuple[Solution, Solution],
    env: LogisticsEnvironment
) -> Solution:
    """
    SREX crossover: Exchange routes between parents
    Simplified version - randomly selects routes from both parents
    """
    parent1, parent2 = parents
    
    if not parent1.routes:
        return parent2.copy()
    if not parent2.routes:
        return parent1.copy()
    
    # Randomly select routes from both parents
    num_routes = random.randint(1, max(1, min(len(parent1.routes), len(parent2.routes))))
    
    offspring_routes = []
    covered_orders = set()
    
    # Take some routes from parent1
    for route in random.sample(parent1.routes, min(num_routes, len(parent1.routes))):
        route_orders = set()
        for step in route['steps']:
            for delivery in step.get('deliveries', []):
                route_orders.add(delivery['order_id'])
        
        if not route_orders.issubset(covered_orders):
            offspring_routes.append(copy.deepcopy(route))
            covered_orders.update(route_orders)
    
    # Fill gaps with routes from parent2
    for route in parent2.routes:
        route_orders = set()
        for step in route['steps']:
            for delivery in step.get('deliveries', []):
                route_orders.add(delivery['order_id'])
        
        if route_orders and not route_orders.issubset(covered_orders):
            offspring_routes.append(copy.deepcopy(route))
            covered_orders.update(route_orders)
    
    return Solution(offspring_routes, env)


# ============================================================================
# PART 7: LOCAL SEARCH (Simplified)
# ============================================================================

class LocalSearch:
    """Local search for solution improvement"""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
    
    def __call__(self, solution: Solution, cost_eval: CostEvaluator) -> Solution:
        """Improve solution via local search"""
        # Simplified: return solution as-is
        # In full implementation: relocate, swap, 2-opt operators
        return solution.copy()


# ============================================================================
# PART 8: GENETIC ALGORITHM
# ============================================================================

@dataclass
class GeneticAlgorithmParams:
    """GA parameters"""
    repair_probability: float = 0.80
    num_iters_no_improvement: int = 100
    max_iterations: int = 200


class GeneticAlgorithm:
    """Hybrid Genetic Search Algorithm"""
    
    def __init__(
        self,
        env: LogisticsEnvironment,
        penalty_manager: PenaltyManager,
        population: Population,
        local_search: LocalSearch,
        initial_solutions: List[Solution],
        params: GeneticAlgorithmParams = GeneticAlgorithmParams()
    ):
        self.env = env
        self.pm = penalty_manager
        self.pop = population
        self.ls = local_search
        self.initial_solutions = initial_solutions
        self.params = params
        self.best = min(initial_solutions, key=lambda s: self.pm.cost_evaluator().cost(s))
    
    def run(self) -> Solution:
        """Run genetic algorithm"""
        print("🧬 Running Hybrid Genetic Search...")
        
        # Initialize population
        cost_eval = self.pm.cost_evaluator()
        for sol in self.initial_solutions:
            self.pop.add(sol, cost_eval)
        
        iters = 0
        iters_no_improvement = 0
        
        while iters < self.params.max_iterations:
            iters += 1
            
            # Restart if no improvement
            if iters_no_improvement >= self.params.num_iters_no_improvement:
                print(f"   Restart at iteration {iters}")
                self.pop.clear()
                for sol in self.initial_solutions:
                    self.pop.add(sol, cost_eval)
                iters_no_improvement = 0
            
            # Selection
            parents = self.pop.select()
            if parents[0] is None or parents[1] is None:
                break
            
            # Crossover
            offspring = selective_route_exchange(parents, self.env)
            
            # Local search
            offspring = self.ls(offspring, cost_eval)
            
            # Add to population
            self.pop.add(offspring, cost_eval)
            self.pm.register(offspring)
            
            # Update best
            curr_best_cost = cost_eval.cost(self.best)
            offspring_cost = cost_eval.cost(offspring)
            
            if offspring_cost < curr_best_cost:
                self.best = offspring.copy()
                iters_no_improvement = 0
                print(f"   Iteration {iters}: New best £{offspring_cost:,.2f}")
            else:
                iters_no_improvement += 1
            
            if iters % 50 == 0:
                print(f"   Iteration {iters}: Best £{curr_best_cost:,.2f}")
        
        print(f"✅ GA complete after {iters} iterations")
        return self.best


# ============================================================================
# PART 9: INITIAL SOLUTION GENERATION
# ============================================================================

class VRPSolverDijkstra:
    """Dijkstra pathfinding with LRU caching (from solver_76)"""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self._build_adjacency_list()
        self.path_cache = OrderedDict()
        self.distance_cache = OrderedDict()
        self.max_cache_size = 10000
        
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
        """Find shortest path using Dijkstra with LRU caching"""
        cache_key = (start, goal)
        
        if cache_key in self.path_cache:
            self.path_cache.move_to_end(cache_key)
            return self.path_cache[cache_key]
        
        pq = [(0, start, [start])]
        visited = set()
        
        while pq:
            dist, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                self._add_to_cache(cache_key, path, len(path) - 1)
                return path
            
            neighbors = self._get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (dist + 1, neighbor, new_path))
        
        return None
    
    def _add_to_cache(self, cache_key, path, distance):
        if len(self.path_cache) >= self.max_cache_size:
            self.path_cache.popitem(last=False)
            self.distance_cache.popitem(last=False)
        self.path_cache[cache_key] = path
        self.distance_cache[cache_key] = distance
    
    def get_path_distance(self, path: List[int]) -> int:
        return len(path) - 1 if path and len(path) > 1 else 0
    
    def get_cached_distance(self, start: int, goal: int) -> Optional[int]:
        cache_key = (start, goal)
        if cache_key in self.distance_cache:
            self.distance_cache.move_to_end(cache_key)
            return self.distance_cache[cache_key]
        return None


class InitialSolutionGenerator:
    """Generate initial solutions using greedy heuristic"""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.vrp_solver = VRPSolverDijkstra(env)
    
    def generate(self, num_solutions: int = 5) -> List[Solution]:
        """Generate diverse initial solutions"""
        print(f"🌱 Generating {num_solutions} initial solutions...")
        solutions = []
        
        # Generate first solution (greedy)
        routes = self._create_greedy_solution()
        if routes:
            solutions.append(Solution(routes, self.env))
            print(f"   Solution 1: {len(routes)} routes")
        
        # Generate variations with randomness
        for i in range(1, num_solutions):
            routes = self._create_greedy_solution(randomness=i * 0.15)
            if routes:
                solutions.append(Solution(routes, self.env))
                print(f"   Solution {i+1}: {len(routes)} routes")
        
        return solutions
    
    def _create_greedy_solution(self, randomness: float = 0.0) -> List[Dict]:
        """Create solution using greedy nearest neighbor heuristic"""
        # Step 1: Allocate inventory
        allocation = self._allocate_inventory()
        
        # Step 2: Create routes from allocation
        routes = []
        used_vehicles = set()  # Track used vehicles to avoid duplication
        
        for wh_id, order_allocations in allocation.items():
            if not order_allocations:
                continue
            
            # Group orders by proximity
            warehouse = self.env.warehouses[wh_id]
            wh_node = warehouse.location.id
            
            # Get available vehicles from this warehouse
            available_vehicles = [v for v in self.env.get_all_vehicles() 
                                 if v.home_warehouse_id == wh_id and v.id not in used_vehicles]
            
            if not available_vehicles:
                continue
            
            # Sort vehicles by cost (cheapest first)
            available_vehicles.sort(key=lambda v: (v.fixed_cost, v.cost_per_km))
            
            # Create routes for each order (simple version - one order per route)
            for order_id, sku_quantities in order_allocations.items():
                order = self.env.orders[order_id]
                order_node = order.destination.id
                
                # Calculate load
                total_weight = 0.0
                total_volume = 0.0
                
                for sku_id, qty in sku_quantities.items():
                    sku = self.env.skus[sku_id]
                    total_weight += sku.weight * qty
                    total_volume += sku.volume * qty
                
                # Select smallest capable vehicle (that hasn't been used)
                selected_vehicle = None
                for vehicle in available_vehicles:
                    if (vehicle.id not in used_vehicles and
                        total_weight <= vehicle.capacity_weight and 
                        total_volume <= vehicle.capacity_volume):
                        selected_vehicle = vehicle
                        used_vehicles.add(vehicle.id)  # Mark as used
                        break
                
                if not selected_vehicle:
                    # No available vehicle - skip this order
                    continue
                
                # Create route
                path = self.vrp_solver.dijkstra_shortest_path(wh_node, order_node)
                if not path:
                    continue
                
                # Build steps
                steps = []
                
                # Step 1: Pickup at warehouse
                pickup_step = {
                    "node_id": wh_node,
                    "pickups": [
                        {"warehouse_id": wh_id, "sku_id": sku_id, "quantity": qty}
                        for sku_id, qty in sku_quantities.items()
                    ],
                    "deliveries": [],
                    "unloads": []
                }
                steps.append(pickup_step)
                
                # Step 2: Travel to delivery (intermediate nodes)
                for node in path[1:-1]:
                    steps.append({
                        "node_id": node,
                        "pickups": [],
                        "deliveries": [],
                        "unloads": []
                    })
                
                # Step 3: Delivery at destination
                delivery_step = {
                    "node_id": order_node,
                    "pickups": [],
                    "deliveries": [
                        {"order_id": order_id, "sku_id": sku_id, "quantity": qty}
                        for sku_id, qty in sku_quantities.items()
                    ],
                    "unloads": []
                }
                steps.append(delivery_step)
                
                # Step 4: Return to warehouse
                path_home = self.vrp_solver.dijkstra_shortest_path(order_node, wh_node)
                if path_home:
                    for node in path_home[1:]:
                        steps.append({
                            "node_id": node,
                            "pickups": [],
                            "deliveries": [],
                            "unloads": []
                        })
                
                route = {
                    "vehicle_id": selected_vehicle.id,
                    "steps": steps
                }
                
                routes.append(route)
        
        return routes
    
    def _allocate_inventory(self) -> Dict:
        """Greedy inventory allocation (simplified from solver_76)"""
        order_ids = self.env.get_all_order_ids()
        warehouse_ids = list(self.env.warehouses.keys())
        
        inventory = {wh_id: self.env.get_warehouse_inventory(wh_id).copy() 
                    for wh_id in warehouse_ids}
        
        allocation = defaultdict(lambda: defaultdict(dict))
        
        for order_id in order_ids:
            requirements = self.env.get_order_requirements(order_id)
            if not requirements:
                continue
            
            customer_node = self.env.get_order_location(order_id)
            if customer_node is None:
                continue
            
            # Find closest warehouse with inventory
            best_wh = None
            best_dist = float('inf')
            
            for wh_id in warehouse_ids:
                wh_node = self.env.warehouses[wh_id].location.id
                
                # Check if warehouse has all required SKUs
                has_all = all(inventory[wh_id].get(sku, 0) >= qty 
                             for sku, qty in requirements.items())
                
                if not has_all:
                    continue
                
                # Calculate distance
                path = self.vrp_solver.dijkstra_shortest_path(wh_node, customer_node)
                if path:
                    dist = self.vrp_solver.get_path_distance(path)
                    if dist < best_dist:
                        best_dist = dist
                        best_wh = wh_id
            
            # Allocate from best warehouse
            if best_wh:
                for sku, qty in requirements.items():
                    allocation[best_wh][order_id][sku] = qty
                    inventory[best_wh][sku] -= qty
        
        return allocation


# ============================================================================
# PART 10: MAIN SOLVER
# ============================================================================

def solver(env: LogisticsEnvironment) -> Dict:
    """
    PyVRP-inspired solver
    
    Returns solution in Robin Logistics format: {"routes": [...]}
    """
    print("🚀 PyVRP-Inspired Solver for Robin Logistics")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Create problem data
    data = ProblemData(env)
    print(f"📊 Problem: {data.num_clients} clients, {data.num_vehicles} vehicles")
    
    # 2. Initialize components
    penalty_manager = PenaltyManager()
    population = Population()
    local_search = LocalSearch(env)
    
    # 3. Generate initial solutions
    print("🌱 Generating initial solutions...")
    init_gen = InitialSolutionGenerator(env)
    initial_solutions = init_gen.generate(num_solutions=10)
    
    if not initial_solutions:
        print("❌ Failed to generate initial solutions")
        return {"routes": []}
    
    print(f"✅ Generated {len(initial_solutions)} initial solutions")
    
    # 4. Run genetic algorithm
    ga = GeneticAlgorithm(
        env,
        penalty_manager,
        population,
        local_search,
        initial_solutions
    )
    
    best_solution = ga.run()
    
    # 5. Return result
    elapsed = time.time() - start_time
    print(f"\n⏱️  Runtime: {elapsed:.2f}s")
    print("=" * 70)
    
    return {"routes": best_solution.routes}


if __name__ == "__main__":
    env = LogisticsEnvironment()
    solution = solver(env)
    
    print(f"\n{'='*60}")
    print(f"📊 PYVRP-STYLE SOLVER RESULTS")
    print(f"{'='*60}")
    print(f"Routes Generated: {len(solution['routes'])}")
    
    is_valid, message = env.validate_solution_business_logic(solution)
    print(f"\n{'✅' if is_valid else '❌'} Validation: {message}")
    
    success, exec_message = env.execute_solution(solution)
    print(f"{'✅' if success else '❌'} Execution: {exec_message}")
    
    if success:
        fully_fulfilled = 0
        for order_id in env.get_all_order_ids():
            status = env.get_order_fulfillment_status(order_id)
            if sum(status['remaining'].values()) == 0:
                fully_fulfilled += 1
        
        stats = env.get_solution_statistics(solution)
        C_you = stats.get('total_cost', 0)
        C_bench = 10000
        fulfillment_pct = (fully_fulfilled / 50) * 100
        S = C_you + C_bench * (100 - fulfillment_pct)
        
        print(f"\n{'='*60}")
        print(f"📈 PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"💰 Total Cost (C_you): £{C_you:,.2f}")
        print(f"📦 Fulfillment: {fully_fulfilled}/50 ({fulfillment_pct:.0f}%)")
        print(f"📏 Total Distance: {stats.get('total_distance', 0):.2f} km")
        print(f"🚚 Vehicles Used: {len(solution['routes'])}")
        print(f"{'='*60}")
        
        fixed_cost = sum(env.get_vehicle_by_id(r['vehicle_id']).fixed_cost 
                        for r in solution['routes'])
        variable_cost = C_you - fixed_cost
        
        print(f"\n💡 Cost Breakdown:")
        print(f"   Fixed Cost: £{fixed_cost:,.2f} ({len(solution['routes'])} vehicles)")
        print(f"   Variable Cost: £{variable_cost:,.2f}")
        
        print(f"\n🎯 Objective Score (S):")
        print(f"   S = £{C_you:,.2f} + £{C_bench:,} × {(100-fulfillment_pct):.1f}%")
        print(f"   S = £{S:,.2f}")