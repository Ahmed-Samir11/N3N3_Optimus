"""
Improved Multi-Scenario DQN Training with Experience Replay
============================================================
Fixes catastrophic forgetting with:
- Experience replay buffer
- Gradual curriculum learning
- Adaptive learning rate
- Detailed per-scenario debugging
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple, Deque
from collections import deque
import random
import time

# ==================================================================
# SIMULATION ENVIRONMENT
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

# ==================================================================
# EXPERIENCE REPLAY BUFFER
# ==================================================================

class ReplayBuffer:
    """Experience replay buffer to prevent catastrophic forgetting"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch from buffer"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int32)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# ==================================================================
# IMPROVED DQN WITH BATCH LEARNING
# ==================================================================

class ImprovedDQN:
    """Enhanced DQN with experience replay and adaptive learning"""
    
    def __init__(self, state_size=5, hidden_layers=[128, 64, 32], action_size=100):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        
        # Xavier initialization
        self.weights = []
        self.biases = []
        
        layer_sizes = [state_size] + hidden_layers + [action_size]
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.learning_rate_min = 0.0001
        self.learning_rate_decay = 0.995
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activation"""
        x = state
        for i in range(len(self.weights) - 1):
            x = np.maximum(0, np.dot(x, self.weights[i]) + self.biases[i])
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        return x
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        return self.forward(state)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Learn from batch of experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Calculate targets
        current_q = np.array([self.forward(s) for s in states])
        next_q = np.array([self.forward(s) for s in next_states])
        
        targets = current_q.copy()
        for i in range(len(states)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Batch update
        total_loss = 0
        for i in range(len(states)):
            loss = self._backprop(states[i], targets[i])
            total_loss += loss
        
        avg_loss = total_loss / len(states)
        return avg_loss
    
    def _backprop(self, state: np.ndarray, target: np.ndarray):
        """Backpropagation with gradient clipping"""
        # Forward pass storing activations
        activations = [state]
        x = state
        
        for i in range(len(self.weights) - 1):
            x = np.maximum(0, np.dot(x, self.weights[i]) + self.biases[i])
            activations.append(x)
        
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        activations.append(output)
        
        # Compute loss
        loss = np.mean((output - target) ** 2)
        
        # Compute gradients (MSE loss)
        delta = 2 * (output - target) / len(target)
        
        # Clip gradients to prevent explosion
        delta = np.clip(delta, -1.0, 1.0)
        
        # Update output layer
        self.weights[-1] -= self.learning_rate * np.outer(activations[-2], delta)
        self.biases[-1] -= self.learning_rate * delta
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T)
            delta[activations[i+1] <= 0] = 0  # ReLU derivative
            delta = np.clip(delta, -1.0, 1.0)
            
            self.weights[i] -= self.learning_rate * np.outer(activations[i], delta)
            self.biases[i] -= self.learning_rate * delta
        
        return loss
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def decay_learning_rate(self):
        """Decay learning rate"""
        if self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_decay
    
    def save(self, filename: str):
        """Save model to JSON"""
        model_data = {
            'state_size': self.state_size,
            'hidden_layers': self.hidden_layers,
            'action_size': self.action_size,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)
        print(f"✅ Model saved to {filename}")
    
    def load(self, filename: str):
        """Load model from JSON"""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        self.epsilon = model_data.get('epsilon', 0.01)
        self.learning_rate = model_data.get('learning_rate', 0.001)
        print(f"✅ Model loaded from {filename}")

# ==================================================================
# SCENARIO PERFORMANCE TRACKER
# ==================================================================

class ScenarioTracker:
    """Track performance across different scenario types"""
    
    def __init__(self):
        self.scenarios = {
            'small': {'episodes': [], 'fulfillment': [], 'rewards': []},   # 10-20 orders
            'medium': {'episodes': [], 'fulfillment': [], 'rewards': []},  # 20-35 orders
            'large': {'episodes': [], 'fulfillment': [], 'rewards': []}    # 35-50 orders
        }
    
    def record(self, scenario_type, episode, fulfillment, reward):
        """Record episode result"""
        self.scenarios[scenario_type]['episodes'].append(episode)
        self.scenarios[scenario_type]['fulfillment'].append(fulfillment)
        self.scenarios[scenario_type]['rewards'].append(reward)
    
    def get_recent_avg(self, scenario_type, last_n=10):
        """Get average of last N episodes for scenario type"""
        data = self.scenarios[scenario_type]
        if len(data['fulfillment']) == 0:
            return 0, 0
        
        recent_fulfillment = data['fulfillment'][-last_n:]
        recent_rewards = data['rewards'][-last_n:]
        
        return np.mean(recent_fulfillment), np.mean(recent_rewards)
    
    def print_summary(self, episode):
        """Print performance summary"""
        print(f"\n{'='*70}")
        print(f"📊 PERFORMANCE SUMMARY - Episode {episode}")
        print(f"{'='*70}")
        
        for scenario_type in ['small', 'medium', 'large']:
            data = self.scenarios[scenario_type]
            if len(data['fulfillment']) > 0:
                avg_fulfill = np.mean(data['fulfillment'][-10:])
                avg_reward = np.mean(data['rewards'][-10:])
                count = len(data['fulfillment'])
                
                print(f"{scenario_type.capitalize():8s} ({count:3d} episodes): "
                      f"Fulfillment={avg_fulfill:6.1%} | Reward={avg_reward:8.1f}")
        
        print(f"{'='*70}\n")

# ==================================================================
# IMPROVED TRAINING WITH DEBUGGING
# ==================================================================

def train_with_curriculum(episodes=500, save_path='dqn_improved.json'):
    """
    Train DQN with:
    - Curriculum learning (gradual difficulty)
    - Experience replay (prevent forgetting)
    - Scenario tracking (debugging)
    """
    
    print("=" * 70)
    print("🚀 IMPROVED MULTI-SCENARIO DQN TRAINING")
    print("=" * 70)
    print("Features:")
    print("  ✅ Experience replay buffer (prevents catastrophic forgetting)")
    print("  ✅ Curriculum learning (gradual difficulty increase)")
    print("  ✅ Adaptive learning rate decay")
    print("  ✅ Per-scenario performance tracking")
    print("=" * 70)
    print()
    
    # Initialize
    dqn = ImprovedDQN(state_size=5, hidden_layers=[128, 64, 32], action_size=100)
    tracker = ScenarioTracker()
    
    best_avg_fulfillment = 0
    best_episode = 0
    
    # Curriculum stages
    def get_curriculum_stage(episode):
        """Determine order count range based on episode"""
        if episode < 150:
            # Stage 1: Small scenarios (master the basics)
            return random.randint(10, 20), 'small'
        elif episode < 350:
            # Stage 2: Medium scenarios (build complexity)
            return random.randint(20, 35), 'medium'
        else:
            # Stage 3: Large scenarios (full complexity)
            return random.randint(35, 50), 'large'
    
    start_time = time.time()
    
    for episode in range(episodes):
        # Get scenario based on curriculum
        num_orders, scenario_type = get_curriculum_stage(episode)
        
        # Create environment
        env = SimulatedVRPEnv(seed=episode, num_orders=num_orders)
        state = env.reset()
        
        episode_reward = 0
        done = False
        steps = 0
        max_steps = num_orders * 3
        
        # Episode loop
        while not done and steps < max_steps:
            # Epsilon-greedy action selection
            if random.random() < dqn.epsilon:
                action = random.randint(0, dqn.action_size - 1)
            else:
                q_values = dqn.predict(state)[0]
                action = np.argmax(q_values)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            dqn.remember(state, action, reward, next_state, done)
            
            # Learn from replay buffer
            if len(dqn.replay_buffer) >= dqn.batch_size:
                loss = dqn.replay()
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Record results
        fulfillment_rate = info.get('fulfillment_rate', 0)
        tracker.record(scenario_type, episode, fulfillment_rate, episode_reward)
        
        # Decay exploration and learning rate
        dqn.decay_epsilon()
        if episode % 50 == 0:
            dqn.decay_learning_rate()
        
        # Progress report every 10 episodes
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            
            # Calculate average across all recent scenarios
            all_recent_fulfill = []
            all_recent_reward = []
            for stype in ['small', 'medium', 'large']:
                fulfill, reward = tracker.get_recent_avg(stype, last_n=10)
                if fulfill > 0:
                    all_recent_fulfill.append(fulfill)
                    all_recent_reward.append(reward)
            
            avg_fulfill = np.mean(all_recent_fulfill) if all_recent_fulfill else 0
            avg_reward = np.mean(all_recent_reward) if all_recent_reward else 0
            
            print(f"Episode {episode+1:3d}/{episodes} | "
                  f"Stage: {scenario_type:6s} ({num_orders:2d} orders) | "
                  f"Avg Fulfill: {avg_fulfill:6.1%} | "
                  f"Avg Reward: {avg_reward:7.1f} | "
                  f"ε={dqn.epsilon:.3f} | "
                  f"LR={dqn.learning_rate:.5f} | "
                  f"Speed: {eps_per_sec:.1f} eps/s")
            
            # Check for improvement
            if avg_fulfill > best_avg_fulfillment:
                best_avg_fulfillment = avg_fulfill
                best_episode = episode + 1
                dqn.save(save_path.replace('.json', '_best.json'))
                print(f"  🎯 NEW BEST! Avg Fulfillment: {avg_fulfill:.1%}")
        
        # Print detailed summary every 50 episodes
        if (episode + 1) % 50 == 0:
            tracker.print_summary(episode + 1)
            dqn.save(save_path.replace('.json', f'_ep{episode+1}.json'))
        
        # Early stopping if performance is excellent
        if episode > 200:
            recent_avg = np.mean([tracker.get_recent_avg(st, 20)[0] 
                                 for st in ['small', 'medium', 'large']])
            if recent_avg > 0.95:
                print(f"\n🎉 EARLY STOPPING! Excellent performance achieved: {recent_avg:.1%}")
                break
    
    # Final save
    dqn.save(save_path)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best Average Fulfillment: {best_avg_fulfillment:.1%} (Episode {best_episode})")
    print(f"Final Model: {save_path}")
    print(f"Best Model: {save_path.replace('.json', '_best.json')}")
    tracker.print_summary(episodes)
    print("=" * 70)
    
    return dqn, tracker


def evaluate_model(model_path: str, test_episodes=30):
    """Comprehensive evaluation on diverse scenarios"""
    
    print(f"\n{'='*70}")
    print(f"🔍 EVALUATING MODEL: {model_path}")
    print(f"{'='*70}\n")
    
    dqn = ImprovedDQN()
    dqn.load(model_path)
    dqn.epsilon = 0  # No exploration during evaluation
    
    results = {
        'small': [],
        'medium': [],
        'large': []
    }
    
    for episode in range(test_episodes):
        for scenario_type in ['small', 'medium', 'large']:
            # Set order count based on scenario type
            if scenario_type == 'small':
                num_orders = random.randint(10, 20)
            elif scenario_type == 'medium':
                num_orders = random.randint(20, 35)
            else:
                num_orders = random.randint(35, 50)
            
            env = SimulatedVRPEnv(seed=10000 + episode, num_orders=num_orders)
            state = env.reset()
            
            done = False
            steps = 0
            max_steps = num_orders * 3
            
            while not done and steps < max_steps:
                q_values = dqn.predict(state)[0]
                action = np.argmax(q_values)
                state, reward, done, info = env.step(action)
                steps += 1
            
            fulfillment = info.get('fulfillment_rate', 0)
            results[scenario_type].append(fulfillment)
    
    # Print results
    print("Evaluation Results:")
    print(f"{'='*70}")
    for scenario_type, fulfillments in results.items():
        avg = np.mean(fulfillments)
        std = np.std(fulfillments)
        min_val = np.min(fulfillments)
        max_val = np.max(fulfillments)
        
        print(f"{scenario_type.capitalize():8s}: {avg:6.1%} ± {std:5.1%} "
              f"(min: {min_val:5.1%}, max: {max_val:5.1%})")
    
    # Overall average
    all_fulfillments = []
    for f_list in results.values():
        all_fulfillments.extend(f_list)
    overall_avg = np.mean(all_fulfillments)
    
    print(f"{'='*70}")
    print(f"Overall Average: {overall_avg:6.1%}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    print("\n🎓 Starting Improved DQN Training with Curriculum Learning\n")
    
    # Train
    trained_dqn, tracker = train_with_curriculum(
        episodes=500,
        save_path='dqn_improved.json'
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("🧪 FINAL EVALUATION")
    print("="*70)
    
    evaluate_model('dqn_improved_best.json', test_episodes=30)
    
    print("\n✅ All done! Use 'dqn_improved_best.json' for your solver.\n")
