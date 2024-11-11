from mazeEnv import MazeEnv
import random
import numpy as np

# Initialize the environment
env = MazeEnv(render_mode="human")

# Initialize the action space and state space
action_size = env.action_space.n
state_size = env.maze.shape[0] * env.maze.shape[1]  # Flattened 2D grid to 1D state size

# Initialize the Q-table with zeros
qtable = np.zeros((state_size, action_size))

# Q-learning parameters
total_episodes = 100        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 0.4                # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.001            # Exponential decay rate for exploration prob

# List of rewards
rewards = []

# Helper function to convert 2D position to 1D state index
def to_state_index(position):
    return position[0] * env.maze.shape[1] + position[1]

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = to_state_index(env.reset())  # Convert the 2D position to 1D state
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_position, reward, done, info = env.step(action)
        new_state = to_state_index(new_position)  # Convert the 2D position to 1D state

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards = total_rewards + reward

        # Our new state is the current state
        state = new_state

        # If done (if the agent reaches the end goal) : finish episode
        env.render()
        if done == True:
            break

    episode += 1
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

env.close()

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)
print(epsilon)