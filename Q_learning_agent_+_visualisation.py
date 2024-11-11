import numpy as np
import random
import matplotlib.pyplot as plt
import os
from mazeEnv import MazeEnv




class QLearningAgent:
    def __init__(self, env, learning_rate=0.8, gamma=0.95, epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01,
                 decay_rate=0.001):
        self.env = env
        self.state_size = env.maze.shape[0] * env.maze.shape[1]  # Flattened state space
        self.action_size = env.action_space.n
        self.qtable = np.zeros((self.state_size, self.action_size))  # Initialize Q-table

        # Q-learning parameters
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Exploration-exploitation parameters
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

        # Tracking rewards
        self.rewards = []
        self.qtable_diff_threshold = 0.001  # Threshold to stop training early
        self.qtable_diff_history = []

    def to_state_index(self, position):
        """Helper function to convert 2D position to 1D state index."""
        return position[0] * self.env.maze.shape[1] + position[1]

    def compute_average_direction(self, q_values):
        """Compute the weighted average direction based on Q-values."""
        # Directions corresponding to each action (right, up, left, down)
        directions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

        # Compute weighted direction based on Q-values
        total_q_value = np.sum(np.abs(q_values))

        if total_q_value == 0:  # If all Q-values are zero
            return None

        weighted_direction = np.dot(q_values, directions) / total_q_value
        return weighted_direction

    def train(self, total_episodes=1000, max_steps=100):
        """Train the agent using Q-learning."""
        prev_qtable = np.copy(self.qtable)

        for episode in range(total_episodes):
            # Reset the environment
            position = self.env.reset()  # Get initial position
            state = self.to_state_index(position)  # Convert the 2D position to 1D state
            done = False
            total_rewards = 0

            for step in range(max_steps):
                # Exploration-exploitation tradeoff
                exp_exp_tradeoff = random.uniform(0, 1)

                # If tradeoff > epsilon -> exploitation
                if exp_exp_tradeoff > self.epsilon:
                    action = np.argmax(self.qtable[state, :])
                # Else explore (random action)
                else:
                    action = self.env.action_space.sample()

                # Take the action and observe the outcome
                new_position, reward, done, info = self.env.step(action)
                new_state = self.to_state_index(new_position)  # Convert to 1D state

                # Q-learning update rule
                self.qtable[state, action] = self.qtable[state, action] + self.learning_rate * (
                        reward + self.gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])

                # Update rewards and state
                total_rewards += reward
                state = new_state

                # Render the environment at each step (ensure pygame window works)
                if self.env.render_mode == "human":
                    self.env.render()

                # If done (goal reached), end the episode
                if done:
                    break

            # Track total rewards per episode
            self.rewards.append(total_rewards)

            # Decay epsilon to reduce exploration over time
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                -self.decay_rate * (episode + 1))

            # Plot Q-table visualization every 20 episodes
            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_rewards}, Epsilon = {self.epsilon:.3f}")
                self.plot_qtable_on_maze(episode)

            # Check Q-table change
            qtable_diff = np.sum(np.abs(self.qtable - prev_qtable))
            self.qtable_diff_history.append(qtable_diff)
            prev_qtable = np.copy(self.qtable)

            # Early stopping if change is minimal
            if episode > 50 and qtable_diff < self.qtable_diff_threshold:
                print(f"Early stopping at episode {episode} due to minimal Q-table changes.")
                agent.plot_average_cumulative_rewards()
                break

    def get_qtable(self):
        """Return the learned Q-table."""
        return self.qtable

    def save_qtable(self, filename="qtable.npy"):
        """Save the Q-table to a file."""
        np.save(filename, self.qtable)

    def load_qtable(self, filename="qtable.npy"):
        """Load the Q-table from a file."""
        self.qtable = np.load(filename)

    def plot_qtable_on_maze(self, episode):
        """Visualize the Q-table arrows on the maze layout using quiver, focusing only on rotation."""
        nrows, ncols = self.env.maze.shape

        folder = 'qtable_plots'
        if not os.path.exists(folder):
            os.makedirs(folder)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot the maze layout: walls (1) will be displayed as black, paths (0) as white
        ax.imshow(self.env.maze, cmap='gray')

        # Movement direction corresponding to each action
        action_to_direction = {
            0: np.array([0, 1]),  # Right
            1: np.array([-1, 0]),  # Up
            2: np.array([0, -1]),  # Left
            3: np.array([1, 0]),  # Down
        }

        # Prepare lists for quiver arrows
        X, Y, U, V = [], [], [], []

        # Draw arrows representing the best action direction at each path location
        for row in range(nrows):
            for col in range(ncols):
                if self.env.maze[row, col] == 1:  # Skip walls
                    continue

                state_index = self.to_state_index((row, col))
                q_values = self.qtable[state_index, :]

                # Skip if all Q-values are zero
                if np.all(q_values == 0):
                    continue

                # Determine the best action (the one with the highest Q-value)
                best_action = np.argmax(q_values)

                # Get the corresponding direction for the best action
                direction = action_to_direction[best_action]

                # Store the arrow's starting point (col, row) and direction (direction)
                X.append(col)
                Y.append(row)
                U.append(direction[1])  # x component (horizontal)
                V.append(direction[0])  # y component (vertical)

        # Plot arrows using quiver (no scaling for consistent arrow lengths)
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='blue')

        ax.set_title('Q-table Visualization on Maze (Best Action Direction)')
        filename = os.path.join(folder, f'qtable_episode_{episode}.png')
        plt.savefig(filename)

        # Clear the figure to avoid overlapping with future plots
        plt.clf()

    def plot_average_cumulative_rewards(self):
        """Plot the average cumulative rewards at the end of training."""

        cumulative_folder = "cumulative_rewards_visualizations"
        if not os.path.exists(cumulative_folder):
            os.makedirs(cumulative_folder)


        cumulative_rewards = np.cumsum(self.rewards)
        average_cumulative_rewards = cumulative_rewards / (np.arange(1, len(self.rewards) + 1))
        plt.plot(average_cumulative_rewards)
        plt.title('Average Cumulative Rewards Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Cumulative Reward')
        plt.grid(True)

        plt.savefig(f"{cumulative_folder}/cumulative_rewards_episode.png")
        plt.close()


# Testing / Running the code
if __name__ == "__main__":
    # Initialize environment
    env = MazeEnv(render_mode="human")

    # Create Q-learning agent
    agent = QLearningAgent(env)

    # Train the agent
    agent.train(total_episodes=1000, max_steps=100)

    # Save the learned Q-table
    # agent.save_qtable()

    # Print the final Q-table
    print("Final Q-table:")
    print(agent.get_qtable())

    # Plot the final Q-table with correctly rotated arrows
    agent.plot_qtable_on_maze()

    # Plot average cumulative rewards
    agent.plot_average_cumulative_rewards()
