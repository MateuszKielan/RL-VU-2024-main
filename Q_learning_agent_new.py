import numpy as np
import random
import matplotlib.pyplot as plt
import os
from mazeEnv import MazeEnv
from utils import display_heatmap, plot_average_cumulative_rewards, plot_learning_stability


#---------------------------------------------------------------------------------------------------------------------------
# MAIN Q-LEARNING ALGORITHM SECTION

class QLearningAgentModified:
    def __init__(self, env, learning_rate=0.8, gamma=0.95, epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.001):
        """
        Initialization function for the Q-learning Algorithm
        
        Parameters.
        - env: The maze environment agent interacts with.
        - learning_rate: The rate at which the agent updates its Q-values.
        - gamma: Discount factor for future rewards.
        - epsilon: Initial probability for the epsilon-greedy strategy to explore.
        - max_epsilon: Maximum value for epsilon at the start of the training.
        - min_epsilon: Minimum value for epsilon.
        - decay_rate: Rate at which epsilon decays.
        """
        self.env = env
        self.state_size = env.maze.shape[0] * env.maze.shape[1]  # number of possible states in the maze
        self.action_size = env.action_space.n  # Get action space from the environment
        self.qtable = np.zeros((self.state_size, self.action_size))  # Initialize Q-table
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Exploration-exploitation parameters
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        # Tracking rewards and cumulative rewards
        self.rewards = []  # to track rewards per episode
        self.cumulative_rewards = []  # to track cumulative rewards for plotting
        
        # Heatmap tracking variables (for visualization)
        self.visit_counts = np.zeros(env.maze.shape)  # To store visit counts
        self.episode_visit_counts = np.zeros(env.maze.shape)  # Counts for the current 20 episodes
        
    def to_state_index(self, position):
        """
        Helper function to convert 2D position to 1D state index.
        
        Parameters:
        - position: current location of the agent in the maze.
        
        Return:
        - integer that represents the agent's position in the maze in 1D form.
        """
        return position[0] * self.env.maze.shape[1] + position[1]
    
    def train(self, total_episodes=2000, max_steps=100, display_interval=20):
        """
        Train the agent using Q-learning and output heatmaps every 'display_interval' episodes.
        
        Parameters:
        - total_episodes: total number of episodes for which the agent will train
        - max_step: the maximum number of steps allowed per episode. After this limit, the episode ends even if the goal is not reached
        - display_interval: the interval at which heatmaps are displayed

        """
        prev_qtable = np.copy(self.qtable)

        for episode in range(total_episodes):
            # Reset the environment at the start of each episode
            state = self.env.reset()
            done = False
            total_rewards = 0
            step = 0
            
            # Reset visit counts for this episode
            episode_visits = np.zeros(self.env.maze.shape)
            
            while not done and step < max_steps:
                step += 1
                
                # Convert state to 2D coordinates (position)
                pos = (state[0], state[1])
                
                # Add the visit to tracking
                episode_visits[pos] += 1
                
                # Take action based on epsilon-greedy strategy
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.qtable[self.to_state_index(pos)])  # Exploit
                
                # Perform the action in the environment
                new_state, reward, done, info = self.env.step(action)
                total_rewards += reward
                
                # Update Q-table
                # Q(s, a) = Q(s, a) + α * [R(s, a) + γ * max(Q(s', a')) - Q(s, a)]
                next_pos = (new_state[0], new_state[1])
                self.qtable[self.to_state_index(pos), action] = self.qtable[self.to_state_index(pos), action] + \
                    self.learning_rate * (reward + self.gamma * np.max(self.qtable[self.to_state_index(next_pos)]) - self.qtable[self.to_state_index(pos), action])
                
                # Transition to the new state
                state = new_state

            # Accumulate episode visits into overall visit counts
            self.episode_visit_counts += episode_visits

            # Store total rewards for this episode
            self.rewards.append(total_rewards)

            # Decay epsilon to encourage exploitation
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

            # After every display_interval episodes, output the heatmap
            if (episode + 1) % display_interval == 0:
                display_heatmap(episode + 1, self.episode_visit_counts, mode="Q")
                self.episode_visit_counts = np.zeros(self.env.maze.shape)  # Reset visit counts
        
        # After training, plot cumulative rewards and learning stability
        plot_average_cumulative_rewards(self.rewards, mode="Q")
        plot_learning_stability(self.rewards, mode="Q")


# End of Main Q-Learning Algorithm
#---------------------------------------------------------------------------------------------------------------------------

# Run the program
if __name__ == "__main__":
    # Initialize environment
    env = MazeEnv(render_mode="human")

    # Create Q-learning agent
    agent = QLearningAgentModified(env)

    # Train the agent
    agent.train(total_episodes=4000, max_steps=100)
