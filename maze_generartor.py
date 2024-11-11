# maze_generator.py

import numpy as np
import random
from collections import deque

def generate_maze(size):
    """
    Generates a maze of the given size that satisfies the specified constraints.
    This function keeps attempting until it finds a valid maze.
    """
    if size < 5:
        size = 5

    while True:
        # Generate a full maze using Recursive Backtracking
        maze = create_full_maze(size)

        # Get all open positions
        open_positions = [(i, j) for i in range(1, size - 1) for j in range(1, size - 1) if maze[i, j] == 0]
        if len(open_positions) < 3:
            continue  # Not enough open spaces, generate a new maze

        # Randomly select start, subgoal, and goal positions
        positions = open_positions.copy()
        random.shuffle(positions)
        start_position = positions.pop()
        sub_goal_position = positions.pop()
        end_goal_position = positions.pop()

        # Place the special positions in the maze
        maze[start_position] = 2  # Start
        maze[sub_goal_position] = 3  # Subgoal
        maze[end_goal_position] = 4  # Goal

        # Check if the maze satisfies the constraints
        if validate_maze(maze, start_position, sub_goal_position, end_goal_position):
            return maze

def create_full_maze(size):
    """
    Creates a full maze using Recursive Backtracking algorithm.
    """
    maze = np.ones((size, size), dtype=int)
    stack = []
    start_cell = (random.randrange(1, size - 1, 2), random.randrange(1, size - 1, 2))
    maze[start_cell] = 0
    stack.append(start_cell)

    while stack:
        current_cell = stack[-1]
        neighbors = get_unvisited_neighbors(current_cell, maze)
        if neighbors:
            next_cell = random.choice(neighbors)
            carve_passage(current_cell, next_cell, maze)
            stack.append(next_cell)
        else:
            stack.pop()
    return maze

def get_unvisited_neighbors(cell, maze):
    """
    Returns unvisited neighboring cells of the given cell.
    """
    x, y = cell
    neighbors = []
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 1 <= nx < maze.shape[0] - 1 and 1 <= ny < maze.shape[1] - 1:
            if maze[nx, ny] == 1:
                neighbors.append((nx, ny))
    return neighbors

def carve_passage(current_cell, next_cell, maze):
    """
    Carves a passage between two cells.
    """
    x1, y1 = current_cell
    x2, y2 = next_cell
    maze[(x1 + x2) // 2, (y1 + y2) // 2] = 0
    maze[x2, y2] = 0

def validate_maze(maze, start, subgoal, goal):
    """
    Validates the maze against the specified constraints.
    """
    # Path from start to goal without passing through subgoal
    path1 = bfs(maze, start, goal, avoid=subgoal)
    if not path1:
        return False

    # Path from start to subgoal
    path2 = bfs(maze, start, subgoal)
    if not path2:
        return False

    # Path from subgoal to goal
    path3 = bfs(maze, subgoal, goal)
    if not path3:
        return False

    # Ensure that path1 does not pass through subgoal
    if subgoal in path1:
        return False

    return True

def bfs(maze, start, goal, avoid=None):
    """
    Breadth-first search to find a path between two points.
    """
    queue = deque()
    queue.append([start])
    visited = set()
    visited.add(start)

    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == goal:
            return path

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if (1 <= nx < maze.shape[0] - 1 and 1 <= ny < maze.shape[1] - 1):
                if maze[nx, ny] != 1 and (nx, ny) not in visited:
                    if avoid and (nx, ny) == avoid:
                        continue
                    visited.add((nx, ny))
                    queue.append(path + [(nx, ny)])
    return None


maze_size = 11  # Use an odd size for better maze generation
maze = generate_maze(maze_size)
if maze is not None:
    print("Generated Maze:")
    print(maze)
else:
    print("Maze generation failed.")
