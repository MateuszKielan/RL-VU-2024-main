# Modified REINFORCE agent with heatmap visualization every 20 episodes
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from mazeEnv import MazeEnv

import numpy as np
import matplotlib.pyplot as plt

# Optimized REINFORCE agent with baseline, reward normalization, and batch processing

# Adding a softmax function to ensure valid probability distributions after policy updates
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp((x - np.max(x))/1.3)  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

class REINFORCEAgentOptimized:
    def __init__(self, env, learning_rate=0.2, gamma=0.97, batch_size=10):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size  # Number of episodes before updating the policy
        
        # Initialize policy: a table of action probabilities (softmax over actions)
        self.policy = np.ones((env.maze.shape[0] * env.maze.shape[1], env.action_space.n)) / env.action_space.n
        
        # Baseline: we'll use a moving average of rewards as the baseline
        self.baseline = 0
        
        # Heatmap tracking variables
        self.episode_visit_counts = np.zeros(env.maze.shape)  # Counts for the current batch of episodes
    
    def to_state_index(self, position):
        """Helper function to convert 2D position to 1D state index."""
        return position[0] * self.env.maze.shape[1] + position[1]
    
    
    def normalize_rewards(self, rewards):
        """Normalize rewards to reduce the scale variability."""
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if np.std(rewards) > 0 else 1
        return (rewards - mean_reward) / std_reward
    
    def get_action(self, state_index, epsilon):
        """Select an action using epsilon-greedy exploration."""
        if np.random.rand() < epsilon:
            # Exploration: choose a random action
            action = self.env.action_space.sample()
        else:
            # Exploitation: follow the policy
            action_probabilities = softmax(self.policy[state_index, :])
            action = np.random.choice(self.env.action_space.n, p=action_probabilities)
        return action

    def train(self, total_episodes=1000, max_steps=100, display_interval=20, epsilon_decay=0.995):
        """Train the agent with epsilon decay."""
        epsilon = 1.0  # Start with full exploration
        accumulated_rewards = []
        accumulated_gradients = []

        for episode in range(total_episodes):
            state = self.env.reset()
            done = False
            step = 0
            rewards = []
            episode_visits = np.zeros(self.env.maze.shape)
            episode_actions = []
            episode_states = []

            while not done and step < max_steps:
                step += 1
                
                # Exploration or exploitation
                state_index = self.to_state_index(state)
                action = self.get_action(state_index, epsilon)
                
                # Perform action
                new_state, reward, done, info = self.env.step(action)
                
                # Store action and rewards
                episode_actions.append(action)
                episode_states.append(state_index)
                rewards.append(reward)
                
                # Track visits
                pos = (state[0], state[1])
                episode_visits[pos] += 1

                # Transition to the next state
                state = new_state
                
                if done:
                    break

            # After each episode, reduce epsilon to decrease exploration
            epsilon = max(0.1, epsilon * epsilon_decay)  # Keep epsilon from going below 0.1

            # Normalize and discount rewards
            discounted_rewards = []
            cumulative_reward = 0
            for reward in reversed(rewards):
                cumulative_reward = reward + self.gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
            
            # Normalize rewards
            discounted_rewards = self.normalize_rewards(discounted_rewards)
            
            # Update baseline using a moving average
            self.baseline = 0.9 * self.baseline + 0.1 * np.mean(discounted_rewards)
            discounted_rewards = np.array(discounted_rewards) - self.baseline  # Subtract baseline

            # Accumulate rewards and gradients
            accumulated_rewards.extend(discounted_rewards)
            accumulated_gradients.append((episode_states, episode_actions, discounted_rewards))
            
            # Accumulate episode visits into overall visit counts
            self.episode_visit_counts += episode_visits
            
            # Update policy in batch after `batch_size` episodes
            if (episode + 1) % self.batch_size == 0:
                self.update_policy_batch(accumulated_gradients)
                accumulated_gradients = []  # Clear the batch
                accumulated_rewards = []  # Reset reward tracking

            # Every 'display_interval' episodes, output the heatmap
            if (episode + 1) % display_interval == 0:
                self.display_heatmap(episode + 1)

                # Reset visit counts for the next batch of episodes
                self.episode_visit_counts = np.zeros(self.env.maze.shape)
                print(f"Episode {episode+1}: Total Reward = {cumulative_reward}")
    def update_policy_batch(self, accumulated_gradients):
        """Update the policy using accumulated gradients from the batch of episodes."""
        for states, actions, rewards in accumulated_gradients:
            for state, action, reward in zip(states, actions, rewards):
                # Softmax gradient update (increase probability of chosen actions based on reward)
                grad = np.zeros(self.policy[state].shape)
                grad[action] = 1
                self.policy[state] = self.policy[state] + self.learning_rate * reward * (grad - self.policy[state])
                self.policy[state] = softmax(self.policy[state])  # Apply softmax to maintain valid probabilities
    
    def display_heatmap(self, episode):
        """Display a heatmap of the most visited paths."""
        plt.figure(figsize=(8, 6))
        plt.imshow(self.episode_visit_counts, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Visit Frequency')
        plt.title(f'Heatmap of Visited Positions - Up to Episode {episode}')
        plt.show()


# This class will now track the visits and display heatmaps every 20 episodes.
# You can integrate this into your existing system and call the `train()` method.
if __name__ == "__main__":
    # Initialize environment
    env = MazeEnv(render_mode="human")

    # Create REINFORCE agent
    reinforce_agent = REINFORCEAgentOptimized(env)

    # Train the agent
    reinforce_agent.train(total_episodes=1000, max_steps=100)

    # After training, you can evaluate the learned policy
    print("Training complete!")