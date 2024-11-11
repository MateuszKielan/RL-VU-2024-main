import numpy as np
import random
import matplotlib.pyplot as plt
from mazeEnv import MazeEnv

class REINFORCEAgent:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate

        # Initialize policy: a table of action probabilities (softmax over actions)
        self.policy = np.ones((env.maze.shape[0] * env.maze.shape[1], env.action_space.n)) / env.action_space.n

    def to_state_index(self, position):
        """Helper function to convert 2D position to 1D state index."""
        return position[0] * self.env.maze.shape[1] + position[1]

    def get_action(self, state_index):
        """Select an action based on the current policy (softmax probabilities)."""
        action_probabilities = self.policy[state_index, :]
        action = np.random.choice(self.env.action_space.n, p=action_probabilities)
        return action

    def train(self, total_episodes=1000, max_steps=100):
        """Train the agent using REINFORCE."""
        for episode in range(total_episodes):
            # Generate an episode
            state = self.to_state_index(self.env.reset())
            episode_data = []  # Store (state, action, reward)

            for step in range(max_steps):
                action = self.get_action(state)
                new_position, reward, done, _ = self.env.step(action)
                new_state = self.to_state_index(new_position)

                episode_data.append((state, action, reward))  # Store the experience
                state = new_state

                if done:
                    break

            # Calculate the returns (discounted rewards) for each state-action pair
            returns = self.compute_discounted_returns([exp[2] for exp in episode_data])

            # Update the policy using the policy gradient theorem
            for i, (state, action, _) in enumerate(episode_data):
                self.update_policy(state, action, returns[i])

            if episode % 20 == 0:
                total_reward = sum([exp[2] for exp in episode_data])
                print(f"Episode {episode}: Total Reward = {total_reward}")

    def compute_discounted_returns(self, rewards):
        """Compute the cumulative discounted rewards (returns) for an episode."""
        returns = np.zeros_like(rewards, dtype=float)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[t]
            returns[t] = cumulative
        return returns

    def update_policy(self, state_index, action, G_t):
        """Update the policy using the gradient ascent rule."""
        # Softmax gradient update
        action_probabilities = self.policy[state_index, :]
        grad_log = -action_probabilities
        grad_log[action] += 1  # Gradient of log π(a|s)

        # Update policy using the returns (G_t)
        self.policy[state_index, :] += self.learning_rate * grad_log * G_t

        # Re-normalize the policy (softmax) to ensure it remains a valid probability distribution
        self.policy[state_index, :] = np.exp(self.policy[state_index, :])  # Exponentiate to keep positive
        self.policy[state_index, :] /= np.sum(self.policy[state_index, :])  # Normalize to sum to 1

# Testing / Running the REINFORCE code
if __name__ == "__main__":
    # Initialize environment
    env = MazeEnv(render_mode="human")

    # Create REINFORCE agent
    reinforce_agent = REINFORCEAgent(env)

    # Train the agent
    reinforce_agent.train(total_episodes=1000, max_steps=100)

    # After training, you can evaluate the learned policy
    print("Training complete!")
