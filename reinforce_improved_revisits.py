
import numpy as np
import random
from mazeEnv import MazeEnv
from utils import display_heatmap, plot_average_cumulative_rewards, plot_learning_stability

#----------------------------------------------------------------------------------------------------------
# MAIN REINFORCE ALGORITHM SECTION 

def softmax(x):
    """
    Compute softmax values for each set of scores in x.

    Parameters:
    - x: set of scores.
    
    Return:
     - e_x: set of softmax values.
    """
    e_x = np.exp((x - np.max(x))/0.25) 
    return e_x / e_x.sum(axis=0)

class REINFORCEAgentOptimized:
    def __init__(self, env, learning_rate=0.01, gamma=0.96, batch_size=10):
        """
        Initialization Function 

        Parameters:
        - 
        """
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate 
        self.batch_size = batch_size  
        
        # Initialize policy: a table of action probabilities (softmax over actions)
        self.policy = np.ones((env.maze.shape[0] * env.maze.shape[1], env.action_space.n)) / env.action_space.n
        
        # Baseline: we'll use a moving average of rewards as the baseline
        self.baseline = 0
        
        # Heatmap tracking variables
        self.episode_visit_counts = np.zeros(env.maze.shape) 
        self.rewards = []  # Track cumulative rewards

    def to_state_index(self, position):
        """Helper function to convert 2D position to 1D state index."""
        return position[0] * self.env.maze.shape[1] + position[1]
    
    def normalize_rewards(self, rewards):
        """Normalize rewards to reduce the scale variability."""
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if np.std(rewards) > 0 else 1
        return (rewards - mean_reward) / std_reward
    
    def get_action(self, state_index, epsilon):
        # Gradually decrease epsilon for exploration
        epsilon = max(0.1, epsilon * 0.85)  
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()  # Exploration
        else:
            action_probabilities = softmax(self.policy[state_index, :])
            action = np.random.choice(self.env.action_space.n, p=action_probabilities)
        return action

    def train(self, total_episodes=2000, max_steps=100, display_interval=20, epsilon_decay=0.99):
        epsilon = 1 
        accumulated_gradients = []
        best_reward = -float('inf')  # Track the best reward for adaptive epsilon decay

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
                
                # Track visits
                pos = (state[0], state[1])
                episode_visits[pos] += 1

                # Reduce penalties to avoid over-restriction
                if episode_visits[pos] > 1:
                    reward -= 1.2  # Less penalty for revisits
                
                # Store action and rewards
                episode_actions.append(action)
                episode_states.append(state_index)
                rewards.append(reward)

                state = new_state
                
                if done:
                    reward += 10
                    break

            epsilon = max(0.1, epsilon * epsilon_decay)

            # Normalize and discount rewards
            discounted_rewards = []
            cumulative_reward = 0
            for reward in reversed(rewards):
                cumulative_reward = reward + self.gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)

            # Apply reward scaling
            discounted_rewards = self.normalize_rewards(discounted_rewards) * 20  # Increase scaling factor to amplify reward effect
            self.baseline = 0.9 * self.baseline + 0.1 * np.mean(discounted_rewards)  # Faster baseline adaptation
            discounted_rewards = np.array(discounted_rewards) - self.baseline  

            # Track cumulative rewards for plotting
            self.rewards.append(np.sum(rewards))

            # Accumulate gradients for batch update
            accumulated_gradients.append((episode_states, episode_actions, discounted_rewards))

            # Track best reward for adaptive epsilon decay
            if np.sum(rewards) > best_reward:
                best_reward = np.sum(rewards)
                epsilon = max(0.1, epsilon * 0.99)  # Slow down decay if agent makes progress

            # Accumulate episode visits into overall visit counts
            self.episode_visit_counts += episode_visits
            
            # Update policy in batch after `batch_size` episodes
            if (episode + 1) % self.batch_size == 0:
                self.update_policy_batch(accumulated_gradients)
                accumulated_gradients = []

            # Every 'display_interval' episodes, output the heatmap
            if (episode + 1) % display_interval == 0:
                display_heatmap(episode + 1, self.episode_visit_counts, mode="R")  # Call display_heatmap from utils
                print(f"Episode {episode+1}: Total Reward = {np.sum(rewards)}")
                # Log or print policy for a specific state
                print(f"Policy for state [0, 0]: {self.policy[self.to_state_index((0, 0)), :]}")  # Example

        # After training, plot average cumulative rewards and learning stability
        plot_average_cumulative_rewards(self.rewards, mode="R")
        plot_learning_stability(self.rewards, mode="R")

    def update_policy_batch(self, accumulated_gradients):
        for states, actions, rewards in accumulated_gradients:
            for state, action, reward in zip(states, actions, rewards):
                grad = np.zeros(self.policy[state].shape)
                grad[action] = 1
                grad -= self.policy[state]
                self.policy[state] = self.policy[state] + self.learning_rate * reward * grad
                self.policy[state] = softmax(self.policy[state])

if __name__ == "__main__":
    env = MazeEnv(render_mode="human")
    reinforce_agent = REINFORCEAgentOptimized(env)
    reinforce_agent.train(total_episodes=2000, max_steps=100)
    print("Training complete!")
