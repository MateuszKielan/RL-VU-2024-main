# File overview
1. mazeEnv.py - In this file, we define the maze environment in a class using the outlined layout of the maze.
2. utils.py - In this file, we define all necessary functions for rendering the episodes and visualising the data about simulations.
3. Q-learning.py - In this file, we define the logic behind Q-learning agent behaviour, contained in one class "QLearningAgentModified" and include the script for testing the agent in a given maze.
4. Reinforce.py - In this file we define the logic behind Reinforce agent behaviour, contained in one class "REINFORCEAgentOptimized", definition of softmax function and include  the script for testing the agent in a given maze.



# Running the files
For an overview of the agent's performance, you can run the files Q-learning.py and Reinforce.py. Both files contain a script at the end, which runs the simulation of learning in human mode (the human mode includes a pygame popup for visualisation of paths that agent takes in each episode)and saves the respective visualisations in separate folders for further inspection. 
