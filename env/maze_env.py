import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MazeEnv(gym.Env):
    def __init__(self, maze):
        super(MazeEnv, self).__init__()
        self.maze = np.array(maze)
        self.n_rows, self.n_cols = self.maze.shape
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)  # Grid of maze size

        # Set starting and goal positions
        self.start_pos = (0, 0)
        self.goal_pos = (self.n_rows - 1, self.n_cols - 1)
        self.current_pos = self.start_pos

    def reset(self):
        self.current_pos = self.start_pos
        return self._get_state()

    def step(self, action):
        next_pos = self._move(action)
        reward = -1  # Default reward for each step
        done = False

        if self.maze[next_pos] == 1:  # Hitting a wall
            next_pos = self.current_pos  # Stay in the same position
        else:
            self.current_pos = next_pos

        if self.current_pos == self.goal_pos:  # Reached the goal
            reward = 100
            done = True

        return self._get_state(), reward, done, {}

    def _move(self, action):
        # Actions: 0 = up, 1 = down, 2 = left, 3 = right
        row, col = self.current_pos
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < self.n_rows - 1:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < self.n_cols - 1:
            col += 1
        return (row, col)

    def _get_state(self):
        return self.current_pos[0] * self.n_cols + self.current_pos[1]

    def render(self):
        """Visualize the maze with the agent's position as '*'."""
        maze_copy = self.maze.copy()
        row, col = self.current_pos

        # Create a visual representation of the maze
        visual_maze = []
        for r in range(self.n_rows):
            visual_row = []
            for c in range(self.n_cols):
                if (r, c) == (row, col):
                    visual_row.append('*')  # Render agent as '*'
                elif self.maze[r][c] == 1:
                    visual_row.append('#')  # Render walls as '#'
                else:
                    visual_row.append(' ')  # Render free space as ' '
            visual_maze.append(visual_row)

        # Print the maze row by row
        for row in visual_maze:
            print(' '.join(row))
        print()  # Blank line for spacing


# Define a simple maze (0 = free space, 1 = wall)
# maze = [
#     [0, 0, 0, 1, 0],
#     [1, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0]
# ]
#
#
#  env = MazeEnv(maze)