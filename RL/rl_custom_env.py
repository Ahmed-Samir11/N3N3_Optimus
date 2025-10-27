"""
Custom RL Environment for Robin Logistics
===================================================

Pure Python implementation of RL environment for training agents
to solve the Multi-Warehouse Vehicle Routing Problem.
"""

import numpy as np
import random
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque


class VRPState:
    """State representation for VRP problem."""
    
    def __init__(self, env_data: Dict):
        """
        Initialize state from environment data.
        
        State includes:
        - Current vehicle assignments
        - Remaining orders
        - Vehicle capacities (remaining)
        - Current locations
        - Partial routes
        """
        self.num_orders = len(env_data['orders'])
        self.num_vehicles = len(env_data['vehicles'])
        self.num_warehouses = len(env_data['warehouses'])
        
        # Track assignments
        self.assigned_orders = set()
        self.vehicle_routes = {v['id']: [] for v in env_data['vehicles']}
        self.vehicle_loads = {v['id']: {'weight': 0, 'volume': 0} for v in env_data['vehicles']}
        self.used_vehicles = set()
        
        # Track remaining capacity
        self.vehicle_capacity = {
            v['id']: {'weight': v['capacity_weight'], 'volume': v['capacity_volume']}
            for v in env_data['vehicles']
        }
        
        # Cache environment data
        self.env_data = env_data
        
    def copy(self):
        """Create deep copy of state."""
        new_state = VRPState.__new__(VRPState)
        new_state.num_orders = self.num_orders
        new_state.num_vehicles = self.num_vehicles
        new_state.num_warehouses = self.num_warehouses
        new_state.assigned_orders = self.assigned_orders.copy()
        new_state.vehicle_routes = {k: v.copy() for k, v in self.vehicle_routes.items()}
        new_state.vehicle_loads = {k: v.copy() for k, v in self.vehicle_loads.items()}
        new_state.used_vehicles = self.used_vehicles.copy()
        new_state.vehicle_capacity = {k: v.copy() for k, v in self.vehicle_capacity.items()}
        new_state.env_data = self.env_data
        return new_state
    
    def to_feature_vector(self) -> np.ndarray:
        """
        Convert state to fixed-size feature vector for RL agent.
        
        Features:
        - Fulfillment rate (% orders assigned)
        - Avg vehicle utilization (weight)
        - Avg vehicle utilization (volume)
        - Number of used vehicles
        - Number of remaining orders
        """
        fulfillment = len(self.assigned_orders) / self.num_orders if self.num_orders > 0 else 0
        
        weight_utils = []
        volume_utils = []
        for v_id, capacity in self.vehicle_capacity.items():
            v_data = next(v for v in self.env_data['vehicles'] if v['id'] == v_id)
            weight_util = 1 - (capacity['weight'] / v_data['capacity_weight'])
            volume_util = 1 - (capacity['volume'] / v_data['capacity_volume'])
            weight_utils.append(weight_util)
            volume_utils.append(volume_util)
        
        avg_weight_util = np.mean(weight_utils) if weight_utils else 0
        avg_volume_util = np.mean(volume_utils) if volume_utils else 0
        num_used = len(self.used_vehicles)
        remaining = self.num_orders - len(self.assigned_orders)
        
        # Normalize features
        features = np.array([
            fulfillment,
            avg_weight_util,
            avg_volume_util,
            num_used / self.num_vehicles,
            remaining / self.num_orders
        ], dtype=np.float32)
        
        return features


class VRPEnvironment:
    """Custom RL Environment for VRP (NO GYM)."""
    
    def __init__(self, robin_env):
        """
        Initialize from robin_logistics environment.
        
        Args:
            robin_env: LogisticsEnvironment instance
        """
        self.robin_env = robin_env
        
        # Extract static data
        self.env_data = self._extract_environment_data()
        
        # Store original inventory for reset
        self.original_inventory = {}
        for wh in self.env_data['warehouses']:
            self.original_inventory[wh['id']] = wh['inventory'].copy()
        
        # State
        self.state = None
        self.episode_reward = 0
        self.step_count = 0
        self.max_steps = 200  # Prevent infinite episodes
        
    def _extract_environment_data(self) -> Dict:
        """Extract environment data once for efficiency."""
        data = {
            'warehouses': [],
            'vehicles': [],
            'orders': [],
            'skus': {}
        }
        
        # Warehouses
        for wh_id, wh in self.robin_env.warehouses.items():
            data['warehouses'].append({
                'id': wh_id,
                'node': wh.location.id,
                'inventory': dict(self.robin_env.get_warehouse_inventory(wh_id))
            })
        
        # Vehicles
        for v in self.robin_env.get_all_vehicles():
            data['vehicles'].append({
                'id': v.id,
                'type': v.type,
                'home_wh': v.home_warehouse_id,
                'capacity_weight': v.capacity_weight,
                'capacity_volume': v.capacity_volume,
                'fixed_cost': v.fixed_cost,
                'var_cost': v.cost_per_km,
                'cost_efficiency': v.fixed_cost / (v.capacity_weight + v.capacity_volume)
            })
        
        # Sort vehicles by efficiency
        data['vehicles'].sort(key=lambda x: x['cost_efficiency'])
        
        # Orders
        for oid in self.robin_env.get_all_order_ids():
            order = self.robin_env.orders[oid]
            reqs = self.robin_env.get_order_requirements(oid)
            
            weight = sum(self.robin_env.skus[sku].weight * qty for sku, qty in reqs.items())
            volume = sum(self.robin_env.skus[sku].volume * qty for sku, qty in reqs.items())
            
            data['orders'].append({
                'id': oid,
                'node': order.destination.id,
                'requirements': dict(reqs),
                'weight': weight,
                'volume': volume
            })
        
        # SKUs
        for sku_id, sku in self.robin_env.skus.items():
            data['skus'][sku_id] = {
                'weight': sku.weight,
                'volume': sku.volume
            }
        
        return data
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Restore warehouse inventory to original state
        for wh in self.env_data['warehouses']:
            wh['inventory'] = self.original_inventory[wh['id']].copy()
        
        self.state = VRPState(self.env_data)
        self.episode_reward = 0
        self.step_count = 0
        return self.state.to_feature_vector()
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.
        
        Action format:
        {
            'type': 'assign_order',  # or 'finish'
            'order_id': 'ORD-1',
            'vehicle_id': 'V-1',
            'warehouse_id': 'WH-1'
        }
        
        Returns:
            next_state: Feature vector
            reward: Float reward
            done: Boolean (episode finished)
            info: Dict with extra info
        """
        self.step_count += 1
        
        # Check if action is to finish
        if action.get('type') == 'finish':
            return self._finish_episode()
        
        # Validate action
        valid, reward, info = self._validate_and_execute_action(action)
        
        if not valid:
            # Invalid action - negative reward
            reward = -10.0
            done = False
            info['invalid'] = True
            return self.state.to_feature_vector(), reward, done, info
        
        # Check if done
        done = (len(self.state.assigned_orders) == self.state.num_orders) or \
               (self.step_count >= self.max_steps)
        
        self.episode_reward += reward
        
        # If done, return final episode reward instead of step reward
        if done:
            return self._finish_episode()
        
        return self.state.to_feature_vector(), reward, done, info
    
    def _validate_and_execute_action(self, action: Dict) -> Tuple[bool, float, Dict]:
        """Validate and execute action, return reward."""
        order_id = action.get('order_id')
        vehicle_id = action.get('vehicle_id')
        warehouse_id = action.get('warehouse_id')
        
        # Check if order already assigned
        if order_id in self.state.assigned_orders:
            return False, -5.0, {'reason': 'order_already_assigned'}
        
        # Get order and vehicle data
        order = next((o for o in self.env_data['orders'] if o['id'] == order_id), None)
        vehicle = next((v for v in self.env_data['vehicles'] if v['id'] == vehicle_id), None)
        warehouse = next((w for w in self.env_data['warehouses'] if w['id'] == warehouse_id), None)
        
        if not (order and vehicle and warehouse):
            return False, -5.0, {'reason': 'invalid_ids'}
        
        # Check warehouse inventory
        for sku, qty in order['requirements'].items():
            if warehouse['inventory'].get(sku, 0) < qty:
                return False, -5.0, {'reason': 'insufficient_inventory'}
        
        # Check vehicle capacity
        rem_weight = self.state.vehicle_capacity[vehicle_id]['weight']
        rem_volume = self.state.vehicle_capacity[vehicle_id]['volume']
        
        if order['weight'] > rem_weight or order['volume'] > rem_volume:
            return False, -5.0, {'reason': 'insufficient_capacity'}
        
        # Valid action - update state
        self.state.assigned_orders.add(order_id)
        self.state.vehicle_routes[vehicle_id].append(order_id)
        self.state.used_vehicles.add(vehicle_id)
        
        # Update capacity
        self.state.vehicle_capacity[vehicle_id]['weight'] -= order['weight']
        self.state.vehicle_capacity[vehicle_id]['volume'] -= order['volume']
        
        # Update warehouse inventory (not restored on rollback for simplicity)
        for sku, qty in order['requirements'].items():
            warehouse['inventory'][sku] -= qty
        
        # Calculate reward
        # Positive for assigning order, bonus for efficiency
        utilization = 1 - (rem_weight - order['weight']) / vehicle['capacity_weight']
        reward = 1.0 + (utilization * 0.5)  # Base + efficiency bonus
        
        return True, reward, {'utilization': utilization}
    
    def _finish_episode(self) -> Tuple[np.ndarray, float, bool, Dict]:
        """Finish episode and calculate final reward."""
        fulfillment = len(self.state.assigned_orders) / self.state.num_orders
        
        # Huge bonus for high fulfillment
        if fulfillment == 1.0:
            final_reward = 100.0  # Complete solution
        elif fulfillment >= 0.95:
            final_reward = 50.0
        elif fulfillment >= 0.90:
            final_reward = 20.0
        else:
            final_reward = -20.0 * (1 - fulfillment)  # Penalty for low fulfillment
        
        # Penalty for using too many vehicles
        vehicle_penalty = -2.0 * len(self.state.used_vehicles)
        
        total_reward = final_reward + vehicle_penalty
        
        info = {
            'fulfillment': fulfillment,
            'num_vehicles': len(self.state.used_vehicles),
            'final_reward': total_reward
        }
        
        return self.state.to_feature_vector(), total_reward, True, info
    
    def get_valid_actions(self) -> List[Dict]:
        """Get list of valid actions from current state."""
        valid_actions = []
        
        # Get unassigned orders
        unassigned = [o for o in self.env_data['orders'] if o['id'] not in self.state.assigned_orders]
        
        for order in unassigned:
            for vehicle in self.env_data['vehicles']:
                # Check capacity
                rem_w = self.state.vehicle_capacity[vehicle['id']]['weight']
                rem_v = self.state.vehicle_capacity[vehicle['id']]['volume']
                
                if order['weight'] <= rem_w and order['volume'] <= rem_v:
                    # Find suitable warehouse
                    for warehouse in self.env_data['warehouses']:
                        has_inventory = all(
                            warehouse['inventory'].get(sku, 0) >= qty
                            for sku, qty in order['requirements'].items()
                        )
                        
                        if has_inventory:
                            valid_actions.append({
                                'type': 'assign_order',
                                'order_id': order['id'],
                                'vehicle_id': vehicle['id'],
                                'warehouse_id': warehouse['id']
                            })
        
        # Don't add 'finish' action - let episode end naturally when:
        # 1. All orders assigned, or
        # 2. No valid actions left (all vehicles full), or  
        # 3. Max steps reached
        
        return valid_actions


class SimpleQLearning:
    """Simple Q-Learning agent (pure Python, no gym)."""
    
    def __init__(self, state_dim: int, learning_rate: float = 0.01, 
                 discount: float = 0.95, epsilon: float = 1.0):
        """Initialize Q-Learning agent."""
        self.state_dim = state_dim
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Q-table (state_features -> action_index -> Q-value)
        # For VRP, we'll use a simpler heuristic-based Q-function
        self.q_table = {}
        
    def get_state_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key."""
        # Discretize state for Q-table
        discretized = tuple((state * 10).astype(int))
        return str(discretized)
    
    def get_q_value(self, state: np.ndarray, action_idx: int) -> float:
        """Get Q-value for state-action pair."""
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = {}
        return self.q_table[key].get(action_idx, 0.0)
    
    def set_q_value(self, state: np.ndarray, action_idx: int, value: float):
        """Set Q-value for state-action pair."""
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = {}
        self.q_table[key][action_idx] = value
    
    def choose_action(self, state: np.ndarray, valid_actions: List[Dict]) -> Tuple[int, Dict]:
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Explore: random action
            idx = random.randint(0, len(valid_actions) - 1)
        else:
            # Exploit: best Q-value
            q_values = [self.get_q_value(state, i) for i in range(len(valid_actions))]
            idx = int(np.argmax(q_values))
        
        return idx, valid_actions[idx]
    
    def update(self, state: np.ndarray, action_idx: int, reward: float, 
               next_state: np.ndarray, next_actions: List[Dict], done: bool):
        """Update Q-value using Q-learning update rule."""
        current_q = self.get_q_value(state, action_idx)
        
        if done:
            target = reward
        else:
            # Max Q-value over next actions
            next_q_values = [self.get_q_value(next_state, i) for i in range(len(next_actions))]
            max_next_q = max(next_q_values) if next_q_values else 0.0
            target = reward + self.gamma * max_next_q
        
        # Q-learning update
        new_q = current_q + self.lr * (target - current_q)
        self.set_q_value(state, action_idx, new_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filepath: str):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)


def train_qlearning(num_episodes: int = 100, save_path: str = 'qlearning_vrp.pkl'):
    """
    Train Q-Learning agent on VRP problem.
    
    Args:
        num_episodes: Number of training episodes
        save_path: Path to save trained model
    """
    from robin_logistics import LogisticsEnvironment
    
    print(f"Training Q-Learning agent for {num_episodes} episodes...")
    print("=" * 80)
    
    # Initialize environment
    robin_env = LogisticsEnvironment()
    env = VRPEnvironment(robin_env)
    
    # Initialize agent
    state_dim = 5  # From VRPState.to_feature_vector()
    agent = SimpleQLearning(state_dim, learning_rate=0.01, discount=0.95, epsilon=1.0)
    
    # Training loop
    episode_rewards = []
    episode_fulfillments = []
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < env.max_steps:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Choose action
            action_idx, action = agent.choose_action(state, valid_actions)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Get next valid actions
            next_valid_actions = env.get_valid_actions() if not done else []
            
            # Update Q-values
            agent.update(state, action_idx, reward, next_state, next_valid_actions, done)
            
            state = next_state
            total_reward += reward
            step += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track metrics
        episode_rewards.append(total_reward)
        fulfillment = len(env.state.assigned_orders) / env.state.num_orders
        episode_fulfillments.append(fulfillment)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_fulfillment = np.mean(episode_fulfillments[-10:])
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Fulfillment: {avg_fulfillment*100:.1f}%, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Save trained agent
    agent.save(save_path)
    print(f"\nTraining complete! Model saved to {save_path}")
    print(f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Final avg fulfillment (last 10): {np.mean(episode_fulfillments[-10:])*100:.1f}%")
    
    return agent, episode_rewards, episode_fulfillments


if __name__ == '__main__':
    # Example: Train Q-Learning agent
    agent, rewards, fulfillments = train_qlearning(num_episodes=50)
    
    print("\nTraining statistics:")
    print(f"  Max reward: {max(rewards):.2f}")
    print(f"  Avg reward: {np.mean(rewards):.2f}")
    print(f"  Max fulfillment: {max(fulfillments)*100:.1f}%")
