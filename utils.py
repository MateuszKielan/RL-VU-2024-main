#-------------------------------------------------------------------------------------------------------------------------------
# Utils file that encompasses functions for rendering and visualizing 
#
# Section 1 - Rendering The Maze Environment
# Section 2 - Visualizing The data (Heatmaps and Plots)
#-------------------------------------------------------------------------------------------------------------------------------

import os
import matplotlib.pyplot as plt
import pygame
import numpy as np

#-------------------------------------------------------------------------------------------------------------------------------
# Section 1 - Rendering the Maze Environment
#-------------------------------------------------------------------------------------------------------------------------------

def init_window(window, window_size):
    """Initialize the PyGame window."""
    if window is None:
        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode((window_size, window_size))
    return window

def render_frame(window, window_size, maze, agent_position, sub_goal_position, end_goal_position, agent_path):
    """Render the environment on a PyGame window."""
    size = maze.shape[0]  
    pix_square_size = window_size / size  # The size of each grid square in pixels

    # Initialize the canvas
    canvas = pygame.Surface((window_size, window_size))
    canvas.fill((255, 255, 255))  # White background

    # Draw the walls, open spaces, and goals
    for i in range(size):
        for j in range(size):
            if maze[i, j] == 1:  # Wall
                color = (0, 0, 0)  # Black
            elif maze[i, j] == 0:  # Open space
                color = (255, 255, 255)  # White
            elif maze[i, j] == 2:  # Start position
                color = (0, 255, 0)  # Green
            elif maze[i, j] == 3:  # Sub-goal
                color = (255, 255, 0)  # Yellow
            elif maze[i, j] == 4:  # End-goal
                color = (255, 0, 0)  # Red
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(j * pix_square_size, i * pix_square_size, pix_square_size, pix_square_size)
            )
    
    # Draw the agent's path
    if len(agent_path) > 1:
        for i in range(len(agent_path) - 1):
            start_pos = ((agent_path[i][1] + 0.5) * pix_square_size, (agent_path[i][0] + 0.5) * pix_square_size)
            end_pos = ((agent_path[i + 1][1] + 0.5) * pix_square_size, (agent_path[i + 1][0] + 0.5) * pix_square_size)
            pygame.draw.line(canvas, (255, 0, 0), start_pos, end_pos, width=5)  # Cyan for the path

    # Add the current agent position to the path if it's not already there
    if len(agent_path) == 0 or not np.array_equal(agent_position, agent_path[-1]):
        agent_path.append(agent_position.copy())

    # Draw the agent
    pygame.draw.circle(
        canvas,
        (0, 0, 255),  # Blue for the agent
        ((agent_position[1] + 0.5) * pix_square_size, (agent_position[0] + 0.5) * pix_square_size),
        pix_square_size / 3
    )

    # Draw the grid lines
    for x in range(size + 1):
        pygame.draw.line(canvas, (0, 0, 0), (0, pix_square_size * x), (window_size, pix_square_size * x), width=3)
        pygame.draw.line(canvas, (0, 0, 0), (pix_square_size * x, 0), (pix_square_size * x, window_size), width=3)

    if window is not None:
        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

def close_window(window):
    """Close the PyGame window."""
    if window is not None:
        pygame.display.quit()
        pygame.quit()

#-------------------------------------------------------------------------------------------------------------------------------
# Section 2 - Visualizing the Data (With Mode Parameter)
#-------------------------------------------------------------------------------------------------------------------------------

def get_directory(mode):
    """Return the directory name based on the mode ('Q' for Q-Learning, 'R' for REINFORCE)."""
    if mode == "Q":
        return "Q-LearningVisualization"
    elif mode == "R":
        return "ReinforceVisualization"
    else:
        raise ValueError("Invalid mode. Use 'Q' for Q-Learning or 'R' for REINFORCE.")

def display_heatmap(episode, episode_visit_counts, mode):
    """
    Method displays a heatmap of the most visited paths.
    
    Parameters:
    - episode: Current episode number for labeling the file.
    - episode_visit_counts: The 2D array tracking visits to each cell.
    - mode: 'Q' for Q-Learning, 'R' for REINFORCE to determine the save directory.
    """
    directory = get_directory(mode)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(episode_visit_counts, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visit Frequency')
    plt.title(f'Heatmap of Visited Positions - Up to Episode {episode}')
    plt.savefig(f"{directory}/heatmap_{episode}.png")
    plt.close()

def plot_average_cumulative_rewards(rewards, mode):
    """
    Method plots the average cumulative rewards over the episodes.
    
    Parameters:
    - rewards: List of rewards per episode.
    - mode: 'Q' for Q-Learning, 'R' for REINFORCE to determine the save directory.
    """
    directory = get_directory(mode)
    
    cumulative_rewards = np.cumsum(rewards)  # Get cumulative sum of rewards
    average_cumulative_rewards = cumulative_rewards / np.arange(1, len(rewards) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(average_cumulative_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.title("Average Cumulative Reward Over Time")
    plt.grid(True)

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}/average_cumulative_rewards.png")
    plt.close()

def plot_learning_stability(rewards, mode, window_size=50):
    """
    Method plots a moving average of rewards to evaluate learning stability.
    
    Parameters:
    - rewards: List of rewards per episode.
    - mode: 'Q' for Q-Learning, 'R' for REINFORCE to determine the save directory.
    - window_size: The size of the window for calculating the moving average.
    """
    directory = get_directory(mode)
    
    moving_avg_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(moving_avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel(f"Moving Average Reward (window={window_size})")
    plt.title("Learning Stability (Moving Average of Cumulative Rewards)")
    plt.grid(True)

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}/learning_stability.png")
    plt.close()
