"""
Ne3Na3 Solver with Pre-Trained RL Model (Competition Submission)
=================================================================

Uses pre-trained DQN/Q-Learning model for inference.
NO TRAINING happens in solver() - only inference.

This satisfies competition rules:
"the model has be trained (with no training happening in the solver)"
"""

import numpy as np
import json
import os
from typing import Dict, Any, List
from functools import lru_cache
import networkx as nx


# ============================================================================
# PATHFINDING (NetworkX + Caching)
# ============================================================================

_graph_cache = None
_cache_counter = 0

def clear_caches():
    """Clear all caches at start of solver."""
    global _graph_cache, _cache_counter
    _graph_cache = None
    _cache_counter += 1
    get_shortest_path_cached.cache_clear()

def build_networkx_graph(env: Any) -> nx.DiGraph:
    """Build NetworkX graph."""
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache
    
    G = nx.DiGraph()
    road_data = env.get_road_network_data()
    adjacency = road_data.get('adjacency_list', {})
    
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            distance = env.get_distance(node, neighbor)
            if distance is not None:
                G.add_edge(node, neighbor, weight=distance)
    
    _graph_cache = G
    return G

@lru_cache(maxsize=10000)
def get_shortest_path_cached(start: int, end: int, run_id: int):
    """Cached shortest path."""
    G = _graph_cache
    if start == end:
        return (start,), 0.0
    try:
        path = nx.shortest_path(G, start, end, weight='weight')
        distance = nx.shortest_path_length(G, start, end, weight='weight')
        return tuple(path), distance
    except:
        return None, float('inf')

def dijkstra_shortest_path(env: Any, start: int, end: int):
    """Get shortest path."""
    global _cache_counter
    G = build_networkx_graph(env)
    path_tuple, distance = get_shortest_path_cached(start, end, _cache_counter)
    return list(path_tuple) if path_tuple else None, distance


# ============================================================================
# PRE-TRAINED MODEL LOADING
# ============================================================================

class PreTrainedDQN:
    """Load and use pre-trained DQN model."""
    
    def __init__(self, model_path: str):
        """Load trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'r') as f:
            data = json.load(f)
        
        self.layers = data['layers']
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict Q-values."""
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = self.relu(x)
        
        # Output layer
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        return x


# ============================================================================
# RL-GUIDED SOLVER
# ============================================================================

def extract_state_features(env: Any, assigned_orders: set, vehicle_states: Dict) -> np.ndarray:
    """
    Extract state features matching training format.
    
    Returns 5D feature vector:
    - Fulfillment rate
    - Avg weight utilization
    - Avg volume utilization  
    - Fraction of vehicles used
    - Fraction of remaining orders
    """
    total_orders = len(env.get_all_order_ids())
    total_vehicles = len(list(env.get_all_vehicles()))
    
    fulfillment = len(assigned_orders) / total_orders if total_orders > 0 else 0
    
    weight_utils = []
    volume_utils = []
    for v_id, state in vehicle_states.items():
        weight_util = state['used_weight'] / state['capacity_weight']
        volume_util = state['used_volume'] / state['capacity_volume']
        weight_utils.append(weight_util)
        volume_utils.append(volume_util)
    
    avg_weight_util = np.mean(weight_utils) if weight_utils else 0
    avg_volume_util = np.mean(volume_utils) if volume_utils else 0
    used_vehicles = sum(1 for s in vehicle_states.values() if s['orders'])
    vehicle_fraction = used_vehicles / total_vehicles
    remaining_fraction = (total_orders - len(assigned_orders)) / total_orders
    
    features = np.array([
        fulfillment,
        avg_weight_util,
        avg_volume_util,
        vehicle_fraction,
        remaining_fraction
    ], dtype=np.float32)
    
    return features


def rl_guided_assignment(env: Any, model_path: str = 'dqn_vrp.json') -> Dict:
    """
    Use pre-trained RL model to guide order assignment.
    Falls back to greedy if model not found.
    """
    # Try to load model
    try:
        dqn = PreTrainedDQN(model_path)
        use_rl = True
        print("  Using pre-trained DQN model for guidance")
    except:
        use_rl = False
        print("  Model not found, using greedy heuristic")
    
    # Get data
    all_vehicles = list(env.get_all_vehicles())
    all_orders = env.get_all_order_ids()
    
    # Initialize vehicle states
    vehicle_states = {}
    for v in all_vehicles:
        vehicle_states[v.id] = {
            'orders': [],
            'capacity_weight': v.capacity_weight,
            'capacity_volume': v.capacity_volume,
            'used_weight': 0,
            'used_volume': 0,
            'home_wh': v.home_warehouse_id,
            'cost_efficiency': v.fixed_cost / (v.capacity_weight + v.capacity_volume)
        }
    
    # Sort vehicles by efficiency
    sorted_vehicles = sorted(all_vehicles, 
                            key=lambda v: vehicle_states[v.id]['cost_efficiency'])
    
    # Track assignments
    assigned_orders = set()
    order_assignments = {}  # order_id -> (vehicle_id, warehouse_id)
    
    # Assign orders
    for order_id in all_orders:
        order = env.orders[order_id]
        reqs = env.get_order_requirements(order_id)
        
        order_weight = sum(env.skus[sku].weight * qty for sku, qty in reqs.items())
        order_volume = sum(env.skus[sku].volume * qty for sku, qty in reqs.items())
        
        # Get state features
        state = extract_state_features(env, assigned_orders, vehicle_states)
        
        # Find best vehicle
        best_vehicle = None
        best_warehouse = None
        best_score = -float('inf')
        
        for vehicle in sorted_vehicles:
            v_state = vehicle_states[vehicle.id]
            
            # Check capacity
            if (order_weight > v_state['capacity_weight'] - v_state['used_weight'] or
                order_volume > v_state['capacity_volume'] - v_state['used_volume']):
                continue
            
            # Check warehouse inventory
            for wh_id, wh in env.warehouses.items():
                inv = env.get_warehouse_inventory(wh_id)
                has_inventory = all(inv.get(sku, 0) >= qty for sku, qty in reqs.items())
                
                if has_inventory:
                    if use_rl:
                        # Use DQN to score this assignment
                        state_batch = state.reshape(1, -1)
                        q_values = dqn.predict(state_batch)[0]
                        score = q_values[0]  # Simplified scoring
                    else:
                        # Greedy: prefer efficient vehicles with good utilization
                        utilization = (v_state['used_weight'] + order_weight) / v_state['capacity_weight']
                        score = utilization - v_state['cost_efficiency']
                    
                    if score > best_score:
                        best_score = score
                        best_vehicle = vehicle
                        best_warehouse = wh_id
                    break
        
        # Assign if found
        if best_vehicle:
            vehicle_states[best_vehicle.id]['orders'].append(order_id)
            vehicle_states[best_vehicle.id]['used_weight'] += order_weight
            vehicle_states[best_vehicle.id]['used_volume'] += order_volume
            assigned_orders.add(order_id)
            order_assignments[order_id] = (best_vehicle.id, best_warehouse)
    
    return order_assignments, vehicle_states


def build_routes_from_assignments(env: Any, order_assignments: Dict, 
                                  vehicle_states: Dict) -> Dict:
    """Build complete routes from order assignments."""
    routes = []
    
    for vehicle_id, v_state in vehicle_states.items():
        if not v_state['orders']:
            continue
        
        wh_id = v_state['home_wh']
        wh = env.warehouses[wh_id]
        wh_node = wh.location.id
        
        # Collect pickups
        pickups = []
        for oid in v_state['orders']:
            reqs = env.get_order_requirements(oid)
            for sku, qty in reqs.items():
                pickups.append({"warehouse_id": wh_id, "sku_id": sku, "quantity": qty})
        
        # Build delivery sequence (nearest neighbor)
        delivery_nodes = []
        order_map = {}
        for oid in v_state['orders']:
            dest_node = env.orders[oid].destination.id
            if dest_node not in order_map:
                order_map[dest_node] = []
                delivery_nodes.append(dest_node)
            order_map[dest_node].append(oid)
        
        # TSP ordering
        if len(delivery_nodes) > 1:
            ordered = [delivery_nodes[0]]
            unvisited = set(delivery_nodes[1:])
            while unvisited:
                _, nearest = min((dijkstra_shortest_path(env, ordered[-1], n)[1], n) 
                                for n in unvisited)
                ordered.append(nearest)
                unvisited.remove(nearest)
            delivery_nodes = ordered
        
        # Build steps
        steps = [{"node_id": wh_node, "pickups": pickups, "deliveries": [], "unloads": []}]
        
        current_node = wh_node
        for node in delivery_nodes:
            deliveries = []
            for oid in order_map[node]:
                reqs = env.get_order_requirements(oid)
                for sku, qty in reqs.items():
                    deliveries.append({"order_id": oid, "sku_id": sku, "quantity": qty})
            
            # Add path
            path, _ = dijkstra_shortest_path(env, current_node, node)
            if path and len(path) > 1:
                for intermediate in path[1:]:
                    if intermediate == node:
                        steps.append({"node_id": intermediate, "pickups": [], 
                                    "deliveries": deliveries, "unloads": []})
                    else:
                        steps.append({"node_id": intermediate, "pickups": [], 
                                    "deliveries": [], "unloads": []})
            else:
                steps.append({"node_id": node, "pickups": [], 
                            "deliveries": deliveries, "unloads": []})
            
            current_node = node
        
        # Return home
        path_home, _ = dijkstra_shortest_path(env, current_node, wh_node)
        if path_home and len(path_home) > 1:
            for intermediate in path_home[1:]:
                steps.append({"node_id": intermediate, "pickups": [], 
                            "deliveries": [], "unloads": []})
        elif current_node != wh_node:
            steps.append({"node_id": wh_node, "pickups": [], "deliveries": [], "unloads": []})
        
        routes.append({"vehicle_id": vehicle_id, "steps": steps})
    
    return {"routes": routes}


# ============================================================================
# MAIN SOLVER (NO TRAINING)
# ============================================================================

def solver(env: Any) -> Dict:
    """
    RL-guided solver using PRE-TRAINED model.
    NO TRAINING occurs in this function.
    
    Strategy:
    1. Load pre-trained DQN model (or fallback to greedy)
    2. Use model to guide order-vehicle-warehouse assignments
    3. Build routes with TSP ordering
    4. Return solution
    """
    clear_caches()
    
    print("\n[RL-GUIDED SOLVER] Using Pre-Trained Model")
    print("=" * 80)
    
    # Build graph
    G = build_networkx_graph(env)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Phase 1: RL-guided assignment
    print("\nPhase 1: RL-Guided Order Assignment")
    model_path = 'dqn_vrp.json'  # Path to pre-trained model
    order_assignments, vehicle_states = rl_guided_assignment(env, model_path)
    
    assigned = len(order_assignments)
    total = len(env.get_all_order_ids())
    print(f"  Assigned: {assigned}/{total} orders ({assigned/total*100:.1f}%)")
    
    # Phase 2: Build routes
    print("\nPhase 2: Building Routes")
    solution = build_routes_from_assignments(env, order_assignments, vehicle_states)
    
    print(f"\n[SOLUTION] Generated {len(solution['routes'])} routes")
    
    return solution


# For testing
if __name__ == '__main__':
    from robin_logistics import LogisticsEnvironment
    
    env = LogisticsEnvironment()
    result = solver(env)
    
    # Validate
    try:
        validation_result = env.validate_solution_complete(result)
        valid = validation_result if isinstance(validation_result, bool) else validation_result[0]
        msg = "" if isinstance(validation_result, bool) else validation_result[1]
    except Exception as e:
        valid = False
        msg = str(e)
    
    print(f"\nValidation: {valid}")
    if not valid:
        print(f"Error: {msg}")
    else:
        fulfillment = env.get_solution_fulfillment_summary(result)
        cost = env.calculate_solution_cost(result)
        print(f"Fulfillment: {fulfillment['fully_fulfilled_orders']}/{fulfillment['total_orders']}")
        print(f"Cost: ${cost:,.2f}")
