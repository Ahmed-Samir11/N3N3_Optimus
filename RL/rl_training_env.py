"""
RL Training Environment Template for MWVRP
===========================================

This is a STARTER TEMPLATE for training RL models offline.
Use this to train Q-Learning, DQN, or Actor-Critic models.

USAGE:
1. Train model offline using this environment
2. Save trained weights
3. Load weights in solver (inference only, no training)
"""

import gym
from gym import spaces
import numpy as np
from robin_logistics import LogisticsEnvironment
from typing import Dict, Tuple
import random


class MWVRPEnvironment(gym.Env):
    """
    Gym-compatible environment for MWVRP.
    
    State: Current partial solution + remaining capacity + inventory
    Action: Assign order to vehicle + warehouse
    Reward: Negative cost + fulfillment bonus - violation penalty
    """
    
    def __init__(self, max_orders=50, max_vehicles=12, max_steps=100):
        super().__init__()
        
        self.max_orders = max_orders
        self.max_vehicles = max_vehicles
        self.max_steps = max_steps
        
        # State space (continuous features)
        # [vehicle_capacities(12*2), inventory(2*3), order_status(50), 
        #  current_cost(1), fulfillment(1)] = ~100 features
        self.state_dim = max_vehicles * 2 + 6 + max_orders + 2
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Action space: (order_idx, vehicle_idx, warehouse_idx)
        # Discrete: 50 orders × 12 vehicles × 2 warehouses = 1200 actions
        self.action_space = spaces.Discrete(max_orders * max_vehicles * 2)
        
        self.env = None
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        # Create new LogisticsEnvironment
        self.env = LogisticsEnvironment()
        
        # Initialize tracking
        self.assigned_orders = set()
        self.used_vehicles = set()
        self.current_solution = {"routes": []}
        self.step_count = 0
        
        # Get initial state
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Convert current environment to feature vector.
        
        Features:
        - Vehicle remaining capacities (normalized)
        - Warehouse inventory levels (normalized)
        - Order assignment status (binary)
        - Current cost (normalized)
        - Fulfillment rate
        """
        state = []
        
        # Vehicle capacities (12 vehicles × 2 features)
        all_vehicles = list(self.env.get_all_vehicles())
        for v in all_vehicles[:self.max_vehicles]:
            if v.id in self.used_vehicles:
                rem_w, rem_v = self.env.get_vehicle_remaining_capacity(v.id)
                state.append(rem_w / v.capacity_weight)  # Normalized
                state.append(rem_v / v.capacity_volume)
            else:
                state.append(1.0)  # Full capacity
                state.append(1.0)
        
        # Warehouse inventory (2 warehouses × 3 SKUs)
        for wh_id in sorted(self.env.warehouses.keys()):
            inv = self.env.get_warehouse_inventory(wh_id)
            for sku in ['Light_Item', 'Medium_Item', 'Heavy_Item']:
                state.append(inv.get(sku, 0) / 120)  # Normalized by max ~120
        
        # Order assignment status (50 orders, binary)
        all_orders = self.env.get_all_order_ids()
        for oid in all_orders[:self.max_orders]:
            state.append(1.0 if oid in self.assigned_orders else 0.0)
        
        # Current cost (normalized by ~$5000 typical max)
        try:
            cost = self.env.calculate_solution_cost(self.current_solution)
            state.append(min(cost / 5000.0, 1.0))
        except:
            state.append(0.0)
        
        # Fulfillment rate
        total_orders = len(all_orders)
        fulfilled = len(self.assigned_orders)
        state.append(fulfilled / total_orders if total_orders > 0 else 0.0)
        
        return np.array(state, dtype=np.float32)
    
    def _decode_action(self, action: int) -> Tuple[str, str, str]:
        """
        Decode discrete action index to (order_id, vehicle_id, warehouse_id).
        
        Action encoding: action = order_idx * 24 + vehicle_idx * 2 + warehouse_idx
        """
        all_orders = self.env.get_all_order_ids()
        all_vehicles = list(self.env.get_all_vehicles())
        warehouses = sorted(self.env.warehouses.keys())
        
        order_idx = action // (self.max_vehicles * 2)
        vehicle_idx = (action % (self.max_vehicles * 2)) // 2
        warehouse_idx = action % 2
        
        order_id = all_orders[order_idx] if order_idx < len(all_orders) else None
        vehicle_id = all_vehicles[vehicle_idx].id if vehicle_idx < len(all_vehicles) else None
        warehouse_id = warehouses[warehouse_idx] if warehouse_idx < len(warehouses) else None
        
        return order_id, vehicle_id, warehouse_id
    
    def _is_action_valid(self, order_id: str, vehicle_id: str, warehouse_id: str) -> bool:
        """Check if action satisfies constraints."""
        if order_id is None or vehicle_id is None or warehouse_id is None:
            return False
        
        if order_id in self.assigned_orders:
            return False  # Already assigned
        
        if vehicle_id in self.used_vehicles:
            return False  # Vehicle already in use
        
        # Check inventory
        req = self.env.get_order_requirements(order_id)
        inv = self.env.get_warehouse_inventory(warehouse_id)
        if not all(inv.get(sku, 0) >= qty for sku, qty in req.items()):
            return False
        
        # Check capacity
        vehicle = next((v for v in self.env.get_all_vehicles() if v.id == vehicle_id), None)
        if vehicle is None:
            return False
        
        total_w = sum(self.env.skus[sku].weight * qty for sku, qty in req.items())
        total_v = sum(self.env.skus[sku].volume * qty for sku, qty in req.items())
        
        if total_w > vehicle.capacity_weight or total_v > vehicle.capacity_volume:
            return False
        
        return True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info).
        """
        self.step_count += 1
        
        # Decode action
        order_id, vehicle_id, warehouse_id = self._decode_action(action)
        
        # Check validity
        if not self._is_action_valid(order_id, vehicle_id, warehouse_id):
            # Invalid action: negative reward
            reward = -10000
            done = False
            info = {"valid": False, "reason": "Invalid action"}
            return self._get_state(), reward, done, info
        
        # Apply action: Create route for this order
        # (Simplified: one order per vehicle for training)
        order = self.env.orders[order_id]
        warehouse = self.env.warehouses[warehouse_id]
        
        req = self.env.get_order_requirements(order_id)
        pickups = [{"warehouse_id": warehouse_id, "sku_id": sku, "quantity": qty} 
                  for sku, qty in req.items()]
        deliveries = [{"order_id": order_id, "sku_id": sku, "quantity": qty} 
                     for sku, qty in req.items()]
        
        # Simplified route (no pathfinding during training for speed)
        route = {
            "vehicle_id": vehicle_id,
            "steps": [
                {"node_id": warehouse.location.id, "pickups": pickups, "deliveries": [], "unloads": []},
                {"node_id": order.destination.id, "pickups": [], "deliveries": deliveries, "unloads": []},
                {"node_id": warehouse.location.id, "pickups": [], "deliveries": [], "unloads": []}
            ]
        }
        
        self.current_solution["routes"].append(route)
        self.assigned_orders.add(order_id)
        self.used_vehicles.add(vehicle_id)
        
        # Calculate reward
        try:
            cost = self.env.calculate_solution_cost(self.current_solution)
            fulfillment_bonus = 1000  # Per order
            reward = -cost + fulfillment_bonus
        except:
            reward = -5000  # Penalty if solution invalid
        
        # Check if done
        total_orders = len(self.env.get_all_order_ids())
        done = (len(self.assigned_orders) >= total_orders) or (self.step_count >= self.max_steps)
        
        # Info
        info = {
            "valid": True,
            "cost": cost if not done else self.env.calculate_solution_cost(self.current_solution),
            "fulfillment": len(self.assigned_orders) / total_orders,
            "assigned_orders": len(self.assigned_orders)
        }
        
        return self._get_state(), reward, done, info
    
    def render(self, mode='human'):
        """Optional: visualize current state."""
        print(f"Step {self.step_count}: {len(self.assigned_orders)}/{self.max_orders} orders assigned")


# ============================================================================
# EXAMPLE: Training with Stable-Baselines3 (PPO)
# ============================================================================

def train_ppo_model():
    """
    Example training script using PPO from stable-baselines3.
    
    Install: pip install stable-baselines3
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    
    # Create environment
    env = MWVRPEnvironment()
    
    # Validate environment
    check_env(env, warn=True)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_mwvrp_tensorboard/"
    )
    
    # Train
    print("Training PPO model...")
    model.learn(total_timesteps=100_000)
    
    # Save
    model.save("mwvrp_ppo_model")
    print("Model saved to mwvrp_ppo_model.zip")
    
    return model


def test_trained_model():
    """Test trained model on new scenario."""
    from stable_baselines3 import PPO
    
    # Load model
    model = PPO.load("mwvrp_ppo_model")
    
    # Test on new environment
    env = MWVRPEnvironment()
    obs = env.reset()
    
    total_reward = 0
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if info.get("valid"):
            print(f"Step: {info['assigned_orders']} orders, Cost: ${info.get('cost', 0):,.0f}")
    
    print(f"\nFinal Reward: {total_reward}")
    print(f"Fulfillment: {info['fulfillment']*100:.1f}%")


# ============================================================================
# EXAMPLE: Custom Actor-Critic Network
# ============================================================================

import torch
import torch.nn as nn

class ActorCriticNetwork(nn.Module):
    """
    Custom Actor-Critic architecture for MWVRP.
    
    Actor: State → Action probabilities
    Critic: State → Value estimate
    """
    
    def __init__(self, state_dim=100, action_dim=1200, hidden_dim=256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Forward pass through network."""
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state, mask=None):
        """Sample action from policy."""
        action_probs, value = self.forward(state)
        
        # Mask invalid actions
        if mask is not None:
            action_probs = action_probs * mask
            action_probs = action_probs / action_probs.sum()
        
        # Sample action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        return action, dist.log_prob(action), value


if __name__ == '__main__':
    print("RL Training Environment Template")
    print("=" * 80)
    print("\nTo train a model:")
    print("  1. Install: pip install stable-baselines3 gym torch")
    print("  2. Run: python rl_training_env.py")
    print("  3. Train offline for several hours/days")
    print("  4. Save weights and use in solver (inference only)")
    print("\n")
    
    # Uncomment to train:
    # model = train_ppo_model()
    
    # Uncomment to test:
    # test_trained_model()
    
    print("Note: Training RL models requires significant compute time.")
    print("For competition, consider hybrid approach: RL + Heuristics")
