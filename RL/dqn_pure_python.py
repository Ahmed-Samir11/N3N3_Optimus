"""
Pure Python Deep Q-Network (DQN) for VRP
=========================================

Neural network implementation using only numpy (no TensorFlow/PyTorch)
For VRP problem with pre-trained model inference.
"""

import numpy as np
import pickle
import json
from typing import List, Tuple, Dict


class NeuralNetwork:
    """Simple feedforward neural network (pure numpy)."""
    
    def __init__(self, layer_sizes: List[int]):
        """
        Initialize neural network.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        """
        self.layers = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases (Xavier initialization)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative."""
        return (x > 0).astype(float)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass.
        
        Returns:
            output: Network output
            activations: List of layer activations (for backprop)
        """
        activations = [x]
        
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = self.relu(x)
            activations.append(x)
        
        # Output layer (no activation)
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        activations.append(x)
        
        return x, activations
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict output (inference only)."""
        output, _ = self.forward(x)
        return output
    
    def backward(self, activations: List[np.ndarray], target: np.ndarray, 
                 learning_rate: float = 0.001):
        """
        Backward pass (gradient descent).
        
        Args:
            activations: List from forward pass
            target: Target output
            learning_rate: Learning rate
        """
        # Calculate output error
        delta = activations[-1] - target
        
        # Backpropagate
        deltas = [delta]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(activations[i])
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)
    
    def save(self, filepath: str):
        """Save network to file."""
        data = {
            'layers': self.layers,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load network from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.layers = data['layers']
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class SimpleDQN:
    """
    Deep Q-Network agent (pure numpy implementation).
    
    Suitable for training and inference on VRP problem.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_layers: List[int] = [64, 32]):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: State feature dimension
            action_dim: Maximum number of actions (approximate)
            hidden_layers: Hidden layer sizes
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create Q-network
        layer_sizes = [state_dim] + hidden_layers + [action_dim]
        self.q_network = NeuralNetwork(layer_sizes)
        self.target_network = NeuralNetwork(layer_sizes)
        self._sync_target_network()
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.target_update_freq = 10  # Episodes
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=5000)
        self.batch_size = 32
        
        # Training stats
        self.episode_count = 0
    
    def _sync_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.weights = [w.copy() for w in self.q_network.weights]
        self.target_network.biases = [b.copy() for b in self.q_network.biases]
    
    def choose_action(self, state: np.ndarray, valid_actions: List[Dict]) -> Tuple[int, Dict]:
        """
        Choose action using epsilon-greedy with DQN Q-values.
        
        Args:
            state: Current state features
            valid_actions: List of valid actions
        
        Returns:
            action_idx: Index of chosen action
            action: Action dict
        """
        if np.random.random() < self.epsilon:
            # Explore
            idx = np.random.randint(0, len(valid_actions))
        else:
            # Exploit: use Q-network
            state_batch = state.reshape(1, -1)
            q_values = self.q_network.predict(state_batch)[0]
            
            # Only consider valid actions
            # For simplicity, we'll use first len(valid_actions) Q-values
            valid_q_values = q_values[:len(valid_actions)]
            idx = int(np.argmax(valid_q_values))
        
        return idx, valid_actions[idx]
    
    def store_experience(self, state: np.ndarray, action_idx: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step (experience replay)."""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Current Q-values
        current_q = self.q_network.predict(states)
        
        # Target Q-values (using target network)
        next_q = self.target_network.predict(next_states)
        max_next_q = np.max(next_q, axis=1)
        
        # Bellman equation - clip actions to valid range
        targets = current_q.copy()
        for i in range(self.batch_size):
            # Clip action index to network output size
            action_idx = min(int(actions[i]), self.action_dim - 1)
            
            if dones[i]:
                targets[i, action_idx] = rewards[i]
            else:
                targets[i, action_idx] = rewards[i] + self.gamma * max_next_q[i]
        
        # Train Q-network
        _, activations = self.q_network.forward(states)
        self.q_network.backward(activations, targets, self.learning_rate)
    
    def end_episode(self):
        """Called at end of episode."""
        self.episode_count += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        if self.episode_count % self.target_update_freq == 0:
            self._sync_target_network()
    
    def save(self, filepath: str):
        """Save trained model."""
        self.q_network.save(filepath)
        print(f"DQN model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model."""
        self.q_network.load(filepath)
        self._sync_target_network()
        self.epsilon = self.epsilon_min  # Use greedy policy for inference
        print(f"DQN model loaded from {filepath}")


def train_dqn(num_episodes: int = 100, save_path: str = 'dqn_vrp.json', 
              early_stopping: bool = True, patience: int = 50):
    """
    Train DQN agent on VRP problem with detailed tracking.
    
    Args:
        num_episodes: Number of training episodes
        save_path: Path to save trained model
        early_stopping: Stop if no improvement for 'patience' episodes
        patience: Number of episodes without improvement before stopping
    """
    from robin_logistics import LogisticsEnvironment
    from rl_custom_env import VRPEnvironment
    
    print(f"Training DQN agent for {num_episodes} episodes...")
    print("=" * 80)
    
    # Initialize environment
    robin_env = LogisticsEnvironment()
    env = VRPEnvironment(robin_env)
    
    # Initialize agent
    state_dim = 50
    action_dim = 1000  # Approximate max actions
    agent = SimpleDQN(state_dim, action_dim, hidden_layers=[512,256,128, 64, 32])
    
    # Detailed tracking
    episode_rewards = []
    episode_fulfillments = []
    episode_steps = []
    episode_invalid_actions = []
    episode_avg_q_values = []
    
    # Best model tracking
    best_fulfillment = 0
    best_episode = 0
    no_improvement_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        invalid_count = 0
        q_values_sum = 0
        q_values_count = 0
        
        while not done and step < env.max_steps:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Choose action and track Q-values
            action_idx, action = agent.choose_action(state, valid_actions)
            
            # Track Q-values (for monitoring learning)
            if agent.epsilon < 0.5:  # Only after exploration phase
                state_batch = state.reshape(1, -1)
                q_vals = agent.q_network.predict(state_batch)[0]
                q_values_sum += np.max(q_vals)
                q_values_count += 1
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Track invalid actions
            if info.get('invalid', False):
                invalid_count += 1
            
            # Store experience
            agent.store_experience(state, action_idx, reward, next_state, done)
            
            # Train
            agent.train_step()
            
            state = next_state
            total_reward += reward
            step += 1
        
        # End episode
        agent.end_episode()
        
        # Track metrics
        episode_rewards.append(total_reward)
        fulfillment = len(env.state.assigned_orders) / env.state.num_orders
        episode_fulfillments.append(fulfillment)
        episode_steps.append(step)
        episode_invalid_actions.append(invalid_count)
        avg_q = q_values_sum / q_values_count if q_values_count > 0 else 0
        episode_avg_q_values.append(avg_q)
        
        # Check for improvement
        if fulfillment > best_fulfillment:
            best_fulfillment = fulfillment
            best_episode = episode
            no_improvement_count = 0
            # Save best model
            agent.save(save_path.replace('.json', '_best.json'))
        else:
            no_improvement_count += 1
        
        # Early stopping check
        if early_stopping and no_improvement_count >= patience:
            print(f"\nEarly stopping at episode {episode+1}")
            print(f"No improvement for {patience} episodes")
            print(f"Best fulfillment: {best_fulfillment*100:.1f}% at episode {best_episode+1}")
            break
        
        # Print detailed progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_fulfillment = np.mean(episode_fulfillments[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            avg_invalid = np.mean(episode_invalid_actions[-10:])
            recent_q = np.mean([q for q in episode_avg_q_values[-10:] if q > 0])
            
            print(f"\nEpisode {episode+1}/{num_episodes}:")
            print(f"  Reward: {avg_reward:.2f} | Fulfillment: {avg_fulfillment*100:.1f}% | Steps: {avg_steps:.1f}")
            print(f"  Invalid actions: {avg_invalid:.1f} | Avg Q-value: {recent_q:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f} | Buffer: {len(agent.memory)} | Best: {best_fulfillment*100:.1f}%")
            
            # Warning signs
            if avg_fulfillment < 0.5 and episode > 20:
                print(f"WARNING: Low fulfillment - agent may be stuck!")
            if avg_invalid > 20:
                print(f"WARNING: Many invalid actions - consider reward shaping")
    
    # Save final model
    agent.save(save_path)
    
    # Final statistics
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Best fulfillment: {best_fulfillment*100:.1f}% (episode {best_episode+1})")
    print(f"Final fulfillment (last 10): {np.mean(episode_fulfillments[-10:])*100:.1f}%")
    print(f"Final reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    
    # Performance analysis
    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Split into phases
    early_phase = episode_fulfillments[:len(episode_fulfillments)//3]
    mid_phase = episode_fulfillments[len(episode_fulfillments)//3:2*len(episode_fulfillments)//3]
    late_phase = episode_fulfillments[2*len(episode_fulfillments)//3:]
    
    print(f"Early phase (eps 1-{len(early_phase)}): Avg fulfillment {np.mean(early_phase)*100:.1f}%")
    print(f"Mid phase (eps {len(early_phase)+1}-{len(early_phase)+len(mid_phase)}): Avg fulfillment {np.mean(mid_phase)*100:.1f}%")
    print(f"Late phase (eps {len(early_phase)+len(mid_phase)+1}-{len(episode_fulfillments)}): Avg fulfillment {np.mean(late_phase)*100:.1f}%")
    
    # Check for overfitting/underfitting
    if np.mean(late_phase) < np.mean(mid_phase):
        print("\nWARNING: Performance degraded in late phase - possible overfitting!")
        print("   Consider: Lower learning rate, more regularization, or use early stopping")
    elif np.mean(late_phase) - np.mean(early_phase) < 0.1:
        print("\nWARNING: Little improvement - possible underfitting!")
        print("   Consider: Longer training, higher learning rate, or reward shaping")
    else:
        print("\nGood learning progression detected")
    
    # Save training history
    history = {
        'rewards': episode_rewards,
        'fulfillments': episode_fulfillments,
        'steps': episode_steps,
        'invalid_actions': episode_invalid_actions,
        'avg_q_values': episode_avg_q_values,
        'best_episode': best_episode,
        'best_fulfillment': best_fulfillment
    }
    
    history_path = save_path.replace('.json', '_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\nTraining history saved to {history_path}")
    
    return agent, history


if __name__ == '__main__':
    # Train DQN agent
    agent, history = train_dqn(num_episodes=500)
    
    print("\n" + "="*80)
    print("FINAL TRAINING SUMMARY")
    print("="*80)
    print(f"\nBest Episode: {history['best_episode']}")
    print(f"Best Fulfillment: {history['best_fulfillment']*100:.1f}%")
    print(f"\nFinal 10 Episodes:")
    print(f"  Avg Reward: {np.mean(history['rewards'][-10:]):.2f}")
    print(f"  Avg Fulfillment: {np.mean(history['fulfillments'][-10:])*100:.1f}%")