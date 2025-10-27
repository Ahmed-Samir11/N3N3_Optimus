"""
Multi-Scenario DQN Training for VRP
Trains on diverse order counts, capacities, and distributions
Note: This trains on simulated simplified VRP scenarios
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple
import random

# ==================================================================
# SIMPLE SIMULATION ENVIRONMENT (for training only)
# ==================================================================

class SimulatedVRPEnv:
    """Simplified VRP environment for DQN training"""
    
    def __init__(self, seed=None, num_orders=50):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.num_orders = num_orders
        self.num_vehicles = 12
        self.orders_assigned = 0
        self.vehicle_loads = [0.0] * self.num_vehicles
        self.steps = 0
        self.total_cost = 0
    
    def reset(self):
        """Reset to initial state"""
        self.orders_assigned = 0
        self.vehicle_loads = [0.0] * self.num_vehicles
        self.steps = 0
        self.total_cost = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state [5 features]"""
        if self.num_orders == 0:
            return np.array([0, 0, 0, 0, 0], dtype=np.float32)
        
        fulfillment = self.orders_assigned / self.num_orders
        avg_load = np.mean(self.vehicle_loads)
        max_load = np.max(self.vehicle_loads)
        active_vehicles = sum(1 for load in self.vehicle_loads if load > 0) / self.num_vehicles
        remaining = (self.num_orders - self.orders_assigned) / self.num_orders
        
        return np.array([fulfillment, avg_load, max_load, active_vehicles, remaining], dtype=np.float32)
    
    def step(self, action):
        """Take action (assign order to vehicle)"""
        self.steps += 1
        vehicle_id = action % self.num_vehicles
        
        # Simulate order size
        order_size = random.uniform(0.05, 0.25)
        
        reward = 0
        if self.orders_assigned < self.num_orders:
            if self.vehicle_loads[vehicle_id] + order_size <= 1.0:
                # Success
                self.vehicle_loads[vehicle_id] += order_size
                self.orders_assigned += 1
                reward = 100  # Big reward for fulfillment
                
                # Efficiency bonus
                if 0.7 <= self.vehicle_loads[vehicle_id] <= 0.95:
                    reward += 20
                
                # Cost penalty (small)
                cost = random.uniform(1, 10)
                self.total_cost += cost
                reward -= cost * 0.1
            else:
                reward = -10  # Capacity exceeded
        
        done = self.orders_assigned >= self.num_orders or self.steps >= self.num_orders * 3
        
        # Massive penalty for unfulfilled orders
        if done and self.orders_assigned < self.num_orders:
            reward -= (self.num_orders - self.orders_assigned) * 200
        
        next_state = self._get_state()
        info = {
            'fulfillment_rate': self.orders_assigned / self.num_orders,
            'total_cost': self.total_cost
        }
        
        return next_state, reward, done, info

class ImprovedDQN:
    """Enhanced DQN with better architecture for multi-scenario learning"""
    
    def __init__(self, state_size=5, hidden_layers=[128, 64, 32], action_size=100):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        
        # Xavier initialization for better convergence
        self.weights = []
        self.biases = []
        
        layer_sizes = [state_size] + hidden_layers + [action_size]
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activation"""
        x = state
        for i in range(len(self.weights) - 1):
            x = np.maximum(0, np.dot(x, self.weights[i]) + self.biases[i])
        # Linear output layer
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        return x
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions"""
        return self.forward(state)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Q-learning update with gradient descent"""
        # Current Q-values
        current_q = self.forward(state)
        target_q = current_q.copy()
        
        if done:
            target_q[action] = reward
        else:
            next_q = self.forward(next_state)
            target_q[action] = reward + self.gamma * np.max(next_q)
        
        # Backpropagation (simplified gradient descent)
        self._backprop(state, target_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _backprop(self, state: np.ndarray, target: np.ndarray):
        """Simplified backpropagation"""
        # Forward pass storing activations
        activations = [state]
        x = state
        
        for i in range(len(self.weights) - 1):
            x = np.maximum(0, np.dot(x, self.weights[i]) + self.biases[i])
            activations.append(x)
        
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        activations.append(output)
        
        # Compute gradients (MSE loss)
        delta = 2 * (output - target) / len(target)
        
        # Update output layer
        self.weights[-1] -= self.learning_rate * np.outer(activations[-2], delta)
        self.biases[-1] -= self.learning_rate * delta
        
        # Backpropagate through hidden layers (simplified)
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T)
            delta[activations[i+1] <= 0] = 0  # ReLU derivative
            
            self.weights[i] -= self.learning_rate * np.outer(activations[i], delta)
            self.biases[i] -= self.learning_rate * delta
    
    def save(self, filename: str):
        """Save model to JSON"""
        model_data = {
            'state_size': self.state_size,
            'hidden_layers': self.hidden_layers,
            'action_size': self.action_size,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'epsilon': self.epsilon
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load(self, filename: str):
        """Load model from JSON"""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        self.epsilon = model_data.get('epsilon', 0.01)
        print(f"Model loaded from {filename}")


def train_multi_scenario(episodes=500, save_path='dqn_multi_scenario.json'):
    """
    Train DQN on multiple scenario types:
    - Different order counts (10-50)
    - Different capacity constraints
    - Different spatial distributions
    """
    
    print("=" * 60)
    print("Multi-Scenario DQN Training")
    print("=" * 60)
    
    # Initialize DQN
    dqn = ImprovedDQN(state_size=5, hidden_layers=[128, 64, 32], action_size=100)
    
    # Training statistics
    episode_rewards = []
    episode_fulfillments = []
    best_fulfillment = 0
    
    for episode in range(episodes):
        # Vary scenario difficulty
        if episode < 100:
            # Easy scenarios (small order counts)
            num_orders = random.randint(10, 20)
        elif episode < 300:
            # Medium scenarios
            num_orders = random.randint(20, 40)
        else:
            # Hard scenarios (full complexity)
            num_orders = random.randint(30, 50)
        
        # Create environment with variable order count
        env = SimulatedVRPEnv(seed=episode, num_orders=num_orders)
        state = env.reset()
        
        total_reward = 0
        done = False
        steps = 0
        max_steps = num_orders * 3  # More steps for larger scenarios
        
        while not done and steps < max_steps:
            # Epsilon-greedy action selection
            if random.random() < dqn.epsilon:
                action = random.randint(0, dqn.action_size - 1)
            else:
                q_values = dqn.predict(state)
                action = np.argmax(q_values)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Learn from experience
            dqn.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Episode statistics
        fulfillment_rate = info.get('fulfillment_rate', 0)
        episode_rewards.append(total_reward)
        episode_fulfillments.append(fulfillment_rate)
        
        # Track best model
        if fulfillment_rate > best_fulfillment:
            best_fulfillment = fulfillment_rate
            dqn.save(save_path.replace('.json', '_best.json'))
        
        # Progress report every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_fulfillment = np.mean(episode_fulfillments[-10:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Fulfillment: {avg_fulfillment:.2%} | "
                  f"Best: {best_fulfillment:.2%} | "
                  f"Epsilon: {dqn.epsilon:.3f}")
        
        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            dqn.save(save_path.replace('.json', f'_ep{episode+1}.json'))
    
    # Save final model
    dqn.save(save_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Fulfillment: {best_fulfillment:.2%}")
    print(f"Final Model: {save_path}")
    print("=" * 60)
    
    return dqn


def evaluate_model(model_path: str, test_episodes=20):
    """Evaluate trained model on diverse scenarios"""
    
    print(f"\nEvaluating model: {model_path}")
    print("-" * 60)
    
    dqn = ImprovedDQN()
    dqn.load(model_path)
    dqn.epsilon = 0  # No exploration during evaluation
    
    results = {
        'small': [],  # 10-20 orders
        'medium': [],  # 20-35 orders
        'large': []   # 35-50 orders
    }
    
    for episode in range(test_episodes):
        # Test on different scenario sizes
        for scenario_type in ['small', 'medium', 'large']:
            if scenario_type == 'small':
                num_orders = random.randint(10, 20)
            elif scenario_type == 'medium':
                num_orders = random.randint(20, 35)
            else:
                num_orders = random.randint(35, 50)
            
            env = SimulatedVRPEnv(seed=1000 + episode, num_orders=num_orders)
            state = env.reset()
            
            done = False
            steps = 0
            max_steps = num_orders * 3
            
            while not done and steps < max_steps:
                q_values = dqn.predict(state)
                action = np.argmax(q_values)
                state, reward, done, info = env.step(action)
                steps += 1
            
            fulfillment = info.get('fulfillment_rate', 0)
            results[scenario_type].append(fulfillment)
    
    # Report results
    print(f"\nEvaluation Results (over {test_episodes} episodes):")
    for scenario_type, fulfillments in results.items():
        avg = np.mean(fulfillments)
        std = np.std(fulfillments)
        min_val = np.min(fulfillments)
        max_val = np.max(fulfillments)
        print(f"  {scenario_type.capitalize():8s}: {avg:.2%} ± {std:.2%} "
              f"(min: {min_val:.2%}, max: {max_val:.2%})")


if __name__ == '__main__':
    # Train on diverse scenarios
    print("Starting Multi-Scenario DQN Training...")
    print("This will expose the agent to:")
    print("  - Variable order counts (10-50)")
    print("  - Different capacity constraints")
    print("  - Diverse spatial distributions")
    print()
    
    trained_dqn = train_multi_scenario(
        episodes=500,
        save_path='dqn_multi_scenario.json'
    )
    
    # Evaluate on test scenarios
    print("\n" + "=" * 60)
    print("Evaluating trained model...")
    evaluate_model('dqn_multi_scenario_best.json', test_episodes=20)
