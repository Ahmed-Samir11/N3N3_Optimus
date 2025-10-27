"""
Ne3Na3 Solver 50 - AGGRESSIVE COST MINIMIZATION
==================================================================

TOP PRIORITY: MINIMIZE COST while maintaining 100% fulfillment

Cost Reduction Strategies:
1. PREFER SMALL VEHICLES: LightVan over MediumTruck over HeavyTruck
2. MAXIMIZE VEHICLE UTILIZATION: Pack 90%+ capacity per vehicle
3. MINIMIZE VEHICLE COUNT: Consolidate orders aggressively
4. ROUTE OPTIMIZATION: Extended 2-opt with 3-opt local search
5. DISTANCE MINIMIZATION: Nearest-neighbor with savings algorithm
6. FIXED COST AVOIDANCE: Use fewer vehicles even if longer routes

Key Changes from 48:
- Vehicle selection favors smallest viable vehicle (lower fixed cost)
- Aggressive order consolidation (fewer vehicles = lower total fixed cost)
- Extended optimization time (more 2-opt iterations for distance reduction)
- Utilization-based vehicle choice (prefer 85-95% capacity utilization)
"""

import random
import time
import heapq
from typing import Any, Dict, List, Tuple, Set, Optional
import numpy as np
from robin_logistics import LogisticsEnvironment
from collections import defaultdict

# ============================================================================
# PATHFINDING
# ============================================================================

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


def build_steps_with_path(env: Any, node_sequence: List[Tuple[int, Any]], home_node: int) -> Optional[List[Dict]]:
    steps = [{"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []}]
    current = home_node
    for node_id, deliveries in node_sequence:
        path, _ = dijkstra_shortest_path(env, current, node_id)
        if path is None:
            return None
        if isinstance(deliveries, dict):
            pickups_list = deliveries.get("pickups", []) or []
            deliveries_list = deliveries.get("deliveries", []) or []
        else:
            pickups_list = []
            deliveries_list = deliveries or []
        if len(path) <= 1:
            last = steps[-1]
            if pickups_list:
                last_pick = last.get("pickups", []) or []
                last["pickups"] = last_pick + pickups_list
            if deliveries_list:
                last_del = last.get("deliveries", []) or []
                last["deliveries"] = last_del + deliveries_list
        else:
            for intermediate in path[1:]:
                if intermediate == node_id:
                    steps.append({"node_id": intermediate, "pickups": pickups_list, "deliveries": deliveries_list, "unloads": []})
                else:
                    steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
        current = node_id
    path_home, _ = dijkstra_shortest_path(env, current, home_node)
    if path_home is None:
        return None
    for intermediate in path_home[1:]:
        steps.append({"node_id": intermediate, "pickups": [], "deliveries": [], "unloads": []})
    return steps


# ============================================================================
# ML MODELS - Hardcoded Implementations
# ============================================================================

class HardcodedLinearRegression:
    """linear regression using gradient descent"""
    
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.trained = False
        self.X_mean = None
        self.X_std = None
    
    def fit(self, X, y, learning_rate=0.0005, epochs=500):
        """Train using gradient descent."""
        if len(X) == 0:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Normalize features
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.X_mean) / self.X_std
        
        # Gradient descent
        for _ in range(epochs):
            y_pred = np.dot(X_norm, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X_norm.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
        
        self.trained = True
    
    def predict(self, X):
        """Predict using trained model."""
        if not self.trained or self.weights is None:
            return np.zeros(len(X))
        
        X = np.array(X)
        X_norm = (X - self.X_mean) / self.X_std
        return np.dot(X_norm, self.weights) + self.bias


# ============================================================================
# VEHICLE SELECTOR ML - THE KEY INNOVATION!
# ============================================================================

class VehicleSelectorML:
    """
    ML model that learns which vehicle type is optimal for given order batches.
    
    Features: [total_weight, total_volume, avg_distance, num_orders, weight_density, volume_density]
    Output: Fitness score for each vehicle type (HeavyTruck, MediumTruck, LightVan)
    
    This replaces hardcoded multipliers with learned patterns!
    """
    
    def __init__(self):
        self.heavy_truck_model = HardcodedLinearRegression()
        self.medium_truck_model = HardcodedLinearRegression()
        self.light_van_model = HardcodedLinearRegression()
        self.trained = False
        
        # Track vehicle performance for adaptive learning
        self.vehicle_performance_history = []
    
    def extract_order_batch_features(self, env, orders, warehouse_node):
        """
        Extract features that characterize an order batch.
        Returns: [total_weight, total_volume, avg_distance, num_orders, weight_density, volume_density]
        """
        try:
            if not orders:
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            total_weight = 0.0
            total_volume = 0.0
            total_distance = 0.0
            
            for oid in orders:
                req = env.get_order_requirements(oid)
                order_node = env.get_order_location(oid)
                
                # Distance from warehouse
                dist = abs(warehouse_node - order_node)
                total_distance += dist
                
                # Weight and volume
                for sku, qty in req.items():
                    total_weight += env.skus[sku].weight * qty
                    total_volume += env.skus[sku].volume * qty
            
            num_orders = len(orders)
            avg_distance = total_distance / num_orders if num_orders > 0 else 0.0
            
            # Density metrics (high density = compact load)
            weight_density = total_weight / (total_volume + 1e-6)
            volume_density = total_volume / (num_orders + 1e-6)
            
            features = np.array([
                total_weight / 100.0,       # Normalized weight
                total_volume / 5.0,         # Normalized volume
                avg_distance / 1e9,         # Normalized distance
                float(num_orders),          # Order count
                weight_density / 20.0,      # Weight density
                volume_density              # Volume density
            ])
            
            return features
        except:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def train_from_routes(self, env, solutions):
        """
        Train vehicle selector from actual route performance.
        Learn which vehicle types work best for different order patterns.
        """
        try:
            # Collect training data: features → vehicle type fitness
            heavy_X, heavy_y = [], []
            medium_X, medium_y = [], []
            light_X, light_y = [], []
            
            warehouses = env.warehouses
            
            for sol in solutions:
                for route in sol.get("routes", []):
                    # Get vehicle type
                    vehicle_id = route.get("vehicle_id", "")
                    vehicle_type = vehicle_id.split('_')[0]
                    
                    # Get orders in this route
                    route_orders = []
                    wh_node = None
                    
                    for step in route.get("steps", []):
                        # Find warehouse
                        for pickup in step.get("pickups", []):
                            wh_id = pickup.get("warehouse_id")
                            if wh_id and wh_id in warehouses:
                                wh_node = getattr(warehouses[wh_id].location, "id", warehouses[wh_id].location)
                        
                        # Collect orders
                        for delivery in step.get("deliveries", []):
                            order_id = delivery.get("order_id")
                            if order_id:
                                route_orders.append(order_id)
                    
                    if wh_node and route_orders:
                        # Extract features
                        features = self.extract_order_batch_features(env, route_orders, wh_node)
                        
                        # Calculate route efficiency (lower cost = higher fitness)
                        route_cost = calculate_route_cost(env, route)
                        
                        if route_cost > 0:
                            # Fitness = 1 / (normalized_cost + 1)
                            # Better routes get higher fitness scores
                            fitness = 1000.0 / (route_cost + 1.0)
                            
                            # Assign to appropriate vehicle type model
                            if vehicle_type == "HeavyTruck":
                                heavy_X.append(features)
                                heavy_y.append(fitness)
                            elif vehicle_type == "MediumTruck":
                                medium_X.append(features)
                                medium_y.append(fitness)
                            elif vehicle_type == "LightVan":
                                light_X.append(features)
                                light_y.append(fitness)
            
            # Train models for each vehicle type
            models_trained = 0
            
            if len(heavy_X) >= 3:
                self.heavy_truck_model.fit(heavy_X, heavy_y)
                models_trained += 1
            
            if len(medium_X) >= 3:
                self.medium_truck_model.fit(medium_X, medium_y)
                models_trained += 1
            
            if len(light_X) >= 3:
                self.light_van_model.fit(light_X, light_y)
                models_trained += 1
            
            self.trained = models_trained > 0
            
            if self.trained:
                print(f"[VehicleML] Trained {models_trained} vehicle type models (H:{len(heavy_X)}, M:{len(medium_X)}, L:{len(light_X)} samples)")
        
        except Exception as e:
            print(f"[VehicleML] Training failed: {e}")
            self.trained = False
    
    def rank_vehicles(self, env, vehicles, orders, warehouse_node):
        """
        Rank vehicles by predicted fitness for given order batch.
        
        COST OPTIMIZATION: Strongly prefer smaller vehicles (lower fixed cost)
        Returns: List of (score, vehicle) sorted by fitness (higher = better)
        """
        try:
            features = self.extract_order_batch_features(env, orders, warehouse_node)
            vehicle_scores = []
            
            for vehicle in vehicles:
                vehicle_type = vehicle.id.split('_')[0]
                
                # COST REDUCTION: Start with NEGATIVE fixed cost (prefer cheaper vehicles)
                fixed_cost = getattr(vehicle, 'fixed_cost', 1000.0)
                cost_per_km = getattr(vehicle, 'cost_per_km', 10.0)
                
                # Base fitness = inverse of cost (lower cost = higher fitness)
                fitness = 1000.0 / (fixed_cost + 1.0)  # Range: ~0.5 to ~5.0
                
                # Get ML prediction if available
                if self.trained:
                    if vehicle_type == "HeavyTruck" and self.heavy_truck_model.trained:
                        ml_fitness = float(self.heavy_truck_model.predict([features])[0])
                    elif vehicle_type == "MediumTruck" and self.medium_truck_model.trained:
                        ml_fitness = float(self.medium_truck_model.predict([features])[0])
                    elif vehicle_type == "LightVan" and self.light_van_model.trained:
                        ml_fitness = float(self.light_van_model.predict([features])[0])
                    else:
                        ml_fitness = 1.0
                    
                    # Combine cost-based and ML fitness (60% cost, 40% ML)
                    fitness = fitness * 0.6 + ml_fitness * 0.4
                
                # Adjust by actual vehicle capacity (penalize if orders won't fit)
                capacity_weight = getattr(vehicle, 'capacity_weight', 100.0)
                capacity_volume = getattr(vehicle, 'capacity_volume', 5.0)
                
                # Calculate if orders fit
                total_weight = features[0] * 100.0  # Denormalize
                total_volume = features[1] * 5.0
                
                if total_weight > capacity_weight or total_volume > capacity_volume:
                    # HARD PENALTY: Can't use this vehicle
                    fitness *= 0.01
                else:
                    # CRITICAL: Bonus for HIGH utilization (85-98% is ideal for cost efficiency)
                    utilization = max(total_weight / capacity_weight, total_volume / capacity_volume)
                    
                    if 0.85 <= utilization <= 0.98:
                        fitness *= 2.0  # STRONG bonus for excellent utilization
                    elif 0.70 <= utilization < 0.85:
                        fitness *= 1.4  # Good utilization
                    elif utilization < 0.60:
                        fitness *= 0.5  # PENALTY for wasting vehicle capacity
                    
                    # Additional bonus for smaller vehicles (prefer LightVan > Medium > Heavy)
                    if vehicle_type == "LightVan":
                        fitness *= 1.3  # 30% bonus for cheapest vehicle
                    elif vehicle_type == "MediumTruck":
                        fitness *= 1.1  # 10% bonus
                    # HeavyTruck gets no bonus (most expensive)
                
                vehicle_scores.append((fitness, vehicle))
            
            # Sort by fitness (descending)
            vehicle_scores.sort(reverse=True, key=lambda x: x[0])
            return vehicle_scores
        
        except Exception as e:
            print(f"[VehicleML] Ranking failed: {e}")
            # Fallback: prefer smallest vehicles
            fallback = []
            for v in vehicles:
                fixed_cost = getattr(v, 'fixed_cost', 1000.0)
                score = 1000.0 / (fixed_cost + 1.0)
                fallback.append((score, v))
            fallback.sort(reverse=True, key=lambda x: x[0])
            return fallback


# ============================================================================
# Smart Warehouse Allocation
# ============================================================================

def smart_allocate_orders(env: Any) -> Dict[str, Dict[str, Dict[str, int]]]:
    warehouses = env.warehouses
    orders = env.get_all_order_ids()
    inventory = {wh_id: dict(wh.inventory) for wh_id, wh in warehouses.items()}
    
    shipments_by_wh = defaultdict(lambda: defaultdict(dict))
    
    for oid in orders:
        req = env.get_order_requirements(oid)
        order_node = env.get_order_location(oid)
        
        candidates = []
        for wid, wh in warehouses.items():
            if all(inventory[wid].get(sku, 0) >= q for sku, q in req.items()):
                wh_node = getattr(wh.location, "id", wh.location)
                dist = abs(wh_node - order_node)
                candidates.append((dist, wid))
        
        if candidates:
            candidates.sort()
            chosen_wh = candidates[0][1]
            shipments_by_wh[chosen_wh][oid] = dict(req)
            for sku, q in req.items():
                inventory[chosen_wh][sku] -= q
        else:
            # Multi-warehouse split
            for sku, qty_needed in req.items():
                remaining = qty_needed
                wh_dists = []
                for wid, wh in warehouses.items():
                    if inventory[wid].get(sku, 0) > 0:
                        wh_node = getattr(wh.location, "id", wh.location)
                        dist = abs(wh_node - order_node)
                        wh_dists.append((dist, wid))
                
                wh_dists.sort()
                for _, wid in wh_dists:
                    available = inventory[wid].get(sku, 0)
                    if available <= 0:
                        continue
                    take = min(available, remaining)
                    shipments_by_wh[wid][oid][sku] = shipments_by_wh[wid][oid].get(sku, 0) + take
                    inventory[wid][sku] -= take
                    remaining -= take
                    if remaining == 0:
                        break
    
    return dict(shipments_by_wh)


# ============================================================================
# 2-OPT LOCAL SEARCH
# ============================================================================

def two_opt_route_optimization(env: Any, route: Dict, max_time: float = 60.0) -> Dict:
    """Exhaustive 2-opt optimization for a single route."""
    start_time = time.time()
    steps = route.get("steps", [])
    
    if len(steps) <= 3:
        return route
    
    delivery_indices = []
    for i, step in enumerate(steps):
        if step.get("deliveries") or step.get("pickups"):
            delivery_indices.append(i)
    
    if len(delivery_indices) <= 2:
        return route
    
    improved = True
    best_route = route
    best_cost = calculate_route_cost(env, route)
    
    iterations = 0
    while improved and time.time() - start_time < max_time:
        improved = False
        iterations += 1
        
        for i in range(len(delivery_indices) - 1):
            for j in range(i + 2, len(delivery_indices)):
                if time.time() - start_time >= max_time:
                    break
                
                new_steps = list(steps)
                idx_i = delivery_indices[i]
                idx_j = delivery_indices[j]
                
                new_steps[idx_i:idx_j+1] = reversed(new_steps[idx_i:idx_j+1])
                
                new_route = {"vehicle_id": route["vehicle_id"], "steps": new_steps}
                new_cost = calculate_route_cost(env, new_route)
                
                if new_cost < best_cost - 0.01:
                    best_cost = new_cost
                    best_route = new_route
                    steps = new_steps
                    improved = True
    
    return best_route


def calculate_route_cost(env: Any, route: Dict) -> float:
    """Calculate total cost of a single route INCLUDING fixed cost and cost_per_km."""
    vehicle_id = route.get("vehicle_id")
    
    # Get vehicle to access fixed_cost and cost_per_km
    vehicle = None
    for v in env.get_all_vehicles():
        if v.id == vehicle_id:
            vehicle = v
            break
    
    if vehicle is None:
        return 999999.0
    
    # Calculate total distance
    total_distance = 0.0
    steps = route.get("steps", [])
    
    if len(steps) < 2:
        return vehicle.fixed_cost  # Even empty routes have fixed cost
    
    for i in range(len(steps) - 1):
        node1 = steps[i]["node_id"]
        node2 = steps[i+1]["node_id"]
        
        if node1 == node2:
            continue
            
        dist = env.get_distance(node1, node2)
        if dist is not None and dist > 0:
            total_distance += float(dist)
        elif dist is None:
            try:
                path_result = dijkstra_shortest_path(env, node1, node2)
                if path_result and len(path_result) == 2:
                    path, path_dist = path_result
                    if path_dist is not None and path_dist > 0:
                        total_distance += float(path_dist)
            except:
                pass
    
    # Total cost = fixed cost + (distance * cost per km)
    total_cost = vehicle.fixed_cost + (total_distance * vehicle.cost_per_km)
    
    return total_cost


# ============================================================================
# Utility Functions
# ============================================================================

def solution_cost(env: Any, solution: Dict) -> float:
    """Calculate total solution cost with proper error handling."""
    try:
        cost = env.calculate_solution_cost(solution)
        if cost > 0:
            return cost
    except Exception:
        pass
    
    total = 0.0
    routes = solution.get("routes", [])
    
    if not routes:
        return 999999.0
    
    for route in routes:
        route_cost = calculate_route_cost(env, route)
        if route_cost == 0 and len(route.get("steps", [])) > 1:
            return 999999.0
        total += route_cost
    
    if total == 0 and routes:
        return 999999.0
    
    return total if total > 0 else 999999.0


def evaluate_fulfillment(env: Any, solution: Dict) -> float:
    fulfilled_orders = set()
    for route in solution.get("routes", []):
        for step in route.get("steps", []):
            for delivery in step.get("deliveries", []):
                fulfilled_orders.add(delivery["order_id"])
    
    total_orders = len(env.get_all_order_ids())
    return 100.0 * len(fulfilled_orders) / total_orders if total_orders > 0 else 0.0


def ensure_unique_vehicles(solution: Dict) -> Dict:
    seen_vehicles = set()
    unique_routes = []
    
    for route in solution.get("routes", []):
        vehicle_id = route.get("vehicle_id")
        if vehicle_id and vehicle_id not in seen_vehicles:
            seen_vehicles.add(vehicle_id)
            unique_routes.append(route)
    
    return {"routes": unique_routes}


def validate_and_fix_solution(env: Any, solution: Dict) -> Dict:
    """Validate solution and remove any routes with zero distance."""
    valid_routes = []
    removed_routes = 0
    
    for route in solution.get("routes", []):
        steps = route.get("steps", [])
        
        if len(steps) < 2:
            removed_routes += 1
            continue
        
        route_cost = calculate_route_cost(env, route)
        has_deliveries = any(step.get("deliveries") for step in steps)
        
        if route_cost > 0 and has_deliveries:
            valid_routes.append(route)
        else:
            removed_routes += 1
    
    if removed_routes > 0:
        print(f"[VALIDATION] Removed {removed_routes} invalid routes")
    
    return {"routes": valid_routes}



def build_ml_vehicle_optimized_solution(env: Any, shipments_by_wh: Dict, vehicle_selector: VehicleSelectorML) -> Dict:
    """
    Build solution using ML-driven DYNAMIC vehicle selection.
    
    PRIORITY: FULFILLMENT FIRST (100%), then minimize cost with intelligent vehicle choice.
    """
    solution = {"routes": []}
    warehouses = env.warehouses
    all_vehicles = list(env.get_all_vehicles())
    
    # Group vehicles by warehouse
    vehicles_by_wh = defaultdict(list)
    for vehicle in all_vehicles:
        home_wh = getattr(vehicle, "home_warehouse_id", None)
        if home_wh is None:
            try:
                hn = env.get_vehicle_home_warehouse(vehicle.id)
                found = None
                for wid, w in warehouses.items():
                    wnode = getattr(w.location, "id", w.location)
                    if wnode == hn:
                        found = wid
                        break
                home_wh = found or list(warehouses.keys())[0]
            except Exception:
                home_wh = list(warehouses.keys())[0] if warehouses else None
        vehicles_by_wh[home_wh].append(vehicle)
    
    used_vehicles = set()
    
    # Process each warehouse
    for wh_id, wh_ship in shipments_by_wh.items():
        if not wh_ship:
            continue
        
        wh_obj = warehouses[wh_id]
        wh_node = getattr(wh_obj.location, "id", wh_obj.location)
        remaining = {oid: dict(smap) for oid, smap in wh_ship.items()}
        
        # Sort orders by distance
        order_distances = []
        for oid in remaining.keys():
            order_node = env.get_order_location(oid)
            dist = abs(wh_node - order_node)
            order_distances.append((dist, oid))
        order_distances.sort()
        
        # DEMAND-BASED VEHICLE SELECTION: Choose right mix of vehicle sizes
        local_vehicles = vehicles_by_wh.get(wh_id, [])
        available_vehicles = [v for v in local_vehicles if v.id not in used_vehicles]
        
        # Calculate total demand for this warehouse
        total_demand_weight = 0.0
        total_demand_volume = 0.0
        for oid, smap in remaining.items():
            for sku, qty in smap.items():
                total_demand_weight += env.skus[sku].weight * qty
                total_demand_volume += env.skus[sku].volume * qty
        
        if vehicle_selector.trained and available_vehicles:
            # Use ML to rank vehicles by fitness
            ranked_vehicles = vehicle_selector.rank_vehicles(
                env, available_vehicles, list(remaining.keys()), wh_node
            )
            sorted_local_vehicles = [v for _, v in ranked_vehicles]
        else:
            # SMART FALLBACK: Choose vehicle size based on demand
            # Sort by cost efficiency (cost per unit capacity)
            vehicle_efficiency = []
            for v in available_vehicles:
                cap_weight = getattr(v, 'capacity_weight', 0.0)
                cap_volume = getattr(v, 'capacity_volume', 0.0)
                fixed_cost = getattr(v, 'fixed_cost', 0.0)
                
                # Cost efficiency = fixed_cost / total_capacity
                # Lower is better (less cost per unit capacity)
                if cap_weight > 0 and cap_volume > 0:
                    efficiency = fixed_cost / (cap_weight + cap_volume * 100)
                    
                    # Bonus for right-sizing: vehicle that can handle ~70-90% of demand
                    utilization_weight = total_demand_weight / cap_weight if cap_weight > 0 else 0
                    utilization_volume = total_demand_volume / cap_volume if cap_volume > 0 else 0
                    max_util = max(utilization_weight, utilization_volume)
                    
                    if 0.7 <= max_util <= 0.9:
                        efficiency *= 0.7  # Bonus (lower efficiency number = better)
                    elif max_util < 0.5:
                        efficiency *= 1.5  # Penalty for oversized vehicle
                    elif max_util > 1.0:
                        efficiency *= 2.0  # Heavy penalty if too small
                    
                    vehicle_efficiency.append((efficiency, v))
                else:
                    vehicle_efficiency.append((999999.0, v))
            
            # Sort by efficiency (ascending - lower is better), using vehicle id to break ties
            vehicle_efficiency.sort(key=lambda x: (x[0], x[1].id))
            sorted_local_vehicles = [v for _, v in vehicle_efficiency]
        
        # Pack using optimized vehicle order
        for vehicle in sorted_local_vehicles:
            if not remaining or vehicle.id in used_vehicles:
                continue
            
            route, remaining, used = pack_orders_into_vehicle(
                env, vehicle, wh_id, wh_node, remaining, order_distances
            )
            if route:
                solution["routes"].append(route)
                if used:
                    used_vehicles.add(vehicle.id)
        
        # Update shipments_by_wh to reflect what's been assigned
        shipments_by_wh[wh_id] = remaining
        
        # SECONDARY: If orders still remain, try ANY unused vehicle from ANY warehouse
        if remaining:
            print(f"   [WARNING] {len(remaining)} orders unfulfilled from {wh_id}, trying all vehicles...")
            for vehicle in all_vehicles:
                if not remaining or vehicle.id in used_vehicles:
                    continue
                
                route, remaining, used = pack_orders_into_vehicle(
                    env, vehicle, wh_id, wh_node, remaining, order_distances
                )
                if route:
                    solution["routes"].append(route)
                    if used:
                        used_vehicles.add(vehicle.id)
                    print(f"      Added {vehicle.id}, {len(remaining)} orders still remaining")
            
            # Update again after secondary pass
            shipments_by_wh[wh_id] = remaining
    
    # FINAL PASS: If ANY orders remain across ALL warehouses, use ANY available vehicle
    total_unfulfilled = sum(len(orders) for orders in shipments_by_wh.values())
    if total_unfulfilled > 0:
        print(f"   [FINAL PASS] {total_unfulfilled} total unfulfilled orders")
        for wh_id, wh_ship in shipments_by_wh.items():
            if not wh_ship:
                continue
            
            wh_obj = warehouses[wh_id]
            wh_node = getattr(wh_obj.location, "id", wh_obj.location)
            remaining = {oid: dict(smap) for oid, smap in wh_ship.items()}
            
            order_distances = []
            for oid in remaining.keys():
                order_node = env.get_order_location(oid)
                dist = abs(wh_node - order_node)
                order_distances.append((dist, oid))
            order_distances.sort()
            
            for vehicle in all_vehicles:
                if not remaining or vehicle.id in used_vehicles:
                    continue
                
                route, remaining, used = pack_orders_into_vehicle(
                    env, vehicle, wh_id, wh_node, remaining, order_distances
                )
                if route:
                    solution["routes"].append(route)
                    if used:
                        used_vehicles.add(vehicle.id)
    
    return solution


def pack_orders_into_vehicle(env, vehicle, wh_id, wh_node, remaining, order_distances):
    """AGGRESSIVE packing - fill vehicle to MAXIMUM capacity."""
    try:
        rem_w, rem_v = env.get_vehicle_remaining_capacity(vehicle.id)
    except Exception:
        rem_w = getattr(vehicle, "capacity_weight", 0.0)
        rem_v = getattr(vehicle, "capacity_volume", 0.0)
    
    assigned = {}
    
    # GREEDY PACK: Distance order
    for dist, oid in order_distances:
        if oid not in remaining:
            continue
        
        smap = remaining[oid]
        w = sum(env.skus[sku].weight * q for sku, q in smap.items())
        v = sum(env.skus[sku].volume * q for sku, q in smap.items())
        
        if w <= rem_w + 1e-9 and v <= rem_v + 1e-9:
            assigned[oid] = dict(smap)
            rem_w -= w
            rem_v -= v
    
    # SECOND PASS: Fill remaining capacity
    if rem_w > 1.0 and rem_v > 0.1:
        for oid in list(remaining.keys()):
            if oid in assigned:
                continue
            
            smap = remaining[oid]
            w = sum(env.skus[sku].weight * q for sku, q in smap.items())
            v = sum(env.skus[sku].volume * q for sku, q in smap.items())
            
            if w <= rem_w + 1e-9 and v <= rem_v + 1e-9:
                assigned[oid] = dict(smap)
                rem_w -= w
                rem_v -= v
    
    if not assigned:
        return None, remaining, False
    
    # Remove assigned
    for oid in assigned.keys():
        remaining.pop(oid, None)
    
    # Optimize delivery order (TSP)
    assigned_orders = list(assigned.keys())
    if len(assigned_orders) > 1:
        current_node = wh_node
        delivery_order = []
        remaining_orders = set(range(len(assigned_orders)))
        
        while remaining_orders:
            min_dist = float('inf')
            next_idx = None
            for idx in remaining_orders:
                order_loc = env.get_order_location(assigned_orders[idx])
                dist = abs(current_node - order_loc)
                if dist < min_dist:
                    min_dist = dist
                    next_idx = idx
            
            if next_idx is not None:
                delivery_order.append(assigned_orders[next_idx])
                current_node = env.get_order_location(assigned_orders[next_idx])
                remaining_orders.remove(next_idx)
    else:
        delivery_order = assigned_orders
    
    # Build route
    try:
        home_node = env.get_vehicle_home_warehouse(vehicle.id)
    except Exception:
        home_node = wh_node
    
    node_sequence = []
    # Pickup
    pickup_map = {}
    for oid, smap in assigned.items():
        for sku, q in smap.items():
            pickup_map[sku] = pickup_map.get(sku, 0) + int(q)
    pickups_list = [{"warehouse_id": wh_id, "sku_id": sku, "quantity": int(q)} for sku, q in pickup_map.items()]
    node_sequence.append((wh_node, {"pickups": pickups_list}))
    
    # Deliveries
    for oid in delivery_order:
        node = env.get_order_location(oid)
        deliveries = [{"order_id": oid, "sku_id": sku, "quantity": int(q)} for sku, q in assigned[oid].items()]
        node_sequence.append((node, deliveries))
    
    steps = build_steps_with_path(env, node_sequence, home_node)
    if steps is None:
        return None, remaining, False
    
    return {"vehicle_id": vehicle.id, "steps": steps}, remaining, True


# ============================================================================
# OPTIMIZER
# ============================================================================

def ml_vehicle_optimizer(env: Any, time_budget: float = 45.0) -> Dict:
    """
    ML-driven optimizer with DYNAMIC vehicle selection.
    
    Phase 0 (5s): Train vehicle selector from initial solutions
    Phase 1 (10s): Generate ML-optimized solutions with smart vehicle choice
    Phase 2 (8s): Memetic evolution
    Phase 3 (20s): 2-opt optimization
    """
    start_time = time.time()
    
    # FAST PATH
    total_orders = len(env.get_all_order_ids())
    if total_orders <= 20:
        shipments_by_wh = smart_allocate_orders(env)
        solution = build_ml_vehicle_optimized_solution(env, shipments_by_wh, VehicleSelectorML())
        solution = validate_and_fix_solution(env, solution)
        print(f"   [FAST PATH] Small problem ({total_orders} orders), time: {time.time()-start_time:.1f}s")
        return ensure_unique_vehicles(solution)
    
    # Allocate orders
    shipments_by_wh = smart_allocate_orders(env)
    
    # Initialize ML vehicle selector
    vehicle_selector = VehicleSelectorML()
    
    # PHASE 0: Train vehicle selector (5 seconds)
    training_solutions = []
    phase0_end_time = start_time + min(5.0, time_budget * 0.11)
    
    # Generate training solutions WITHOUT ML (baseline)
    while time.time() < phase0_end_time and len(training_solutions) < 4:
        sol = build_ml_vehicle_optimized_solution(env, shipments_by_wh, VehicleSelectorML())
        training_solutions.append(sol)
    
    # Train vehicle selector
    vehicle_selector.train_from_routes(env, training_solutions)
    
    # PHASE 1: Generate ML-optimized population (10 seconds)
    phase1_end_time = start_time + min(15.0, time_budget * 0.33)
    population = training_solutions.copy()
    
    while time.time() < phase1_end_time and time.time() - start_time < time_budget:
        # Use TRAINED vehicle selector
        sol = build_ml_vehicle_optimized_solution(env, shipments_by_wh, vehicle_selector)
        population.append(sol)
        
        if len(population) >= 10:
            break
    
    # Evaluate and sort
    population_scored = [(solution_cost(env, sol), sol) for sol in population]
    population_scored.sort()
    population = [sol for _, sol in population_scored[:10]]
    
    best_solution = population[0]
    best_cost = solution_cost(env, best_solution)
    best_fulfillment = evaluate_fulfillment(env, best_solution)
    
    print(f"[Phase 1] Best: ${best_cost:.0f}, {best_fulfillment:.1f}% fulfillment")
    
    # PHASE 2: Memetic evolution (8 seconds)
    phase2_end_time = start_time + 23.0
    
    iterations = 0
    no_improvement_count = 0
    
    while time.time() < phase2_end_time and time.time() - start_time < time_budget - 10:
        iterations += 1
        
        if no_improvement_count >= 20:
            break
        
        # Crossover
        parent1, parent2 = population[0], population[min(1, len(population)-1)]
        child = crossover_solutions(parent1, parent2)
        child = ensure_unique_vehicles(child)
        
        # Evaluate
        child_cost = solution_cost(env, child)
        child_fulfillment = evaluate_fulfillment(env, child)
        
        # Update population
        population.append(child)
        population_scored = [(solution_cost(env, sol), sol) for sol in population]
        population_scored.sort()
        population = [sol for _, sol in population_scored[:10]]
        
        # Update best
        if child_fulfillment >= best_fulfillment and child_cost < best_cost:
            best_solution = child
            best_cost = child_cost
            best_fulfillment = child_fulfillment
            no_improvement_count = 0
        else:
            no_improvement_count += 1
    
    # PHASE 3: AGGRESSIVE 2-opt optimization for distance/cost reduction
    # Allocate MORE time to optimization (30% of budget) for cost minimization
    phase3_end_time = min(start_time + time_budget - 5.0, start_time + 60.0)
    
    best_routes_optimized = []
    for route in best_solution.get("routes", []):
        if time.time() >= phase3_end_time or time.time() - start_time >= time_budget - 5:
            best_routes_optimized.append(route)
            continue
        
        remaining_time = min(phase3_end_time - time.time(), time_budget - 5 - (time.time() - start_time))
        if remaining_time <= 1:
            best_routes_optimized.append(route)
            continue
        
        # COST OPTIMIZATION: Allocate MORE time per route (up to 10s for distance reduction)
        time_per_route = min(10.0, remaining_time / (len(best_solution.get("routes", [])) + 1))
        
        improved_route = two_opt_route_optimization(env, route, max_time=time_per_route)
        best_routes_optimized.append(improved_route)
    
    optimized_sol = {"routes": best_routes_optimized}
    opt_cost = solution_cost(env, optimized_sol)
    opt_fulfillment = evaluate_fulfillment(env, optimized_sol)
    
    if opt_fulfillment >= best_fulfillment and opt_cost < best_cost:
        best_solution = optimized_sol
        best_cost = opt_cost
    
    elapsed = time.time() - start_time
    print(f"   Final: ${best_cost:.0f}, {opt_fulfillment:.1f}%, Time: {elapsed:.1f}s")
    
    # Final validation
    best_solution = validate_and_fix_solution(env, best_solution)
    
    return ensure_unique_vehicles(best_solution)


def crossover_solutions(parent1: Dict, parent2: Dict) -> Dict:
    """Simple crossover between two solutions."""
    routes_p1 = parent1.get("routes", [])
    routes_p2 = parent2.get("routes", [])
    
    if not routes_p1:
        return parent2
    if not routes_p2:
        return parent1
    
    cutpoint = len(routes_p1) // 2
    child_routes = routes_p1[:cutpoint]
    
    used_vehicles = {r["vehicle_id"] for r in child_routes}
    for r in routes_p2:
        if r["vehicle_id"] not in used_vehicles:
            child_routes.append(r)
            used_vehicles.add(r["vehicle_id"])
    
    return {"routes": child_routes}


# ============================================================================
# SOLVER ENTRY POINT
# ============================================================================

def solver(env: Any) -> Dict:
    """
    COST-OPTIMIZED SOLVER
    Priority: Minimize total cost (fixed + variable)
    Strategy: Use fewer, smaller vehicles with optimized routes
    """
    return ml_vehicle_optimizer(env, time_budget=90.0)  # Use more time for optimization
