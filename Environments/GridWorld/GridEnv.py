# import all dependencies
import numpy as np
import gym
from gym import error, spaces, utils
import copy
from gym.utils import seeding

class GridEnv(gym.Env):
    
    def __init__(self, size=5,
                 obstacleMap = [[0,0,0,0,0], [0,0,0,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,0,0,0]],
                 waterMap = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,1,0,0]]):
        # Define the Action Space
        self.action_space = spaces.Discrete(4)
        self.n_actions = 4
        self.actions = [0, 1, 2, 3] # Encoding for values ["up", "down", "left", "right"]

        # Define the transition probabilities
        self.success_rate = 0.9
        self.prob_veerHorizontal = 0
        
        # Define the Observation Space
        self.n_states = size*size
        self.observation_space = spaces.Discrete(self.n_states)
        self.state = 0 # Start State is always zero
        self.terminal_state = self.n_states - 1
        
        # Define the Grid
        self.grid_size = size
        self.grid = np.zeros([size,size])
        
        # Put in obstacles, denoted by value 1 in Grid
        for i in range(size):
            for j in range(size):
                if(obstacleMap[i][j]==1):
                    self.grid[i][j] = 1

        self.waterMap = waterMap
                    
        self.done = False

        self.optimalValueFunction = \
            [[4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
             [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
             [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
             [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
             [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]]
                    
        
    def step(self, action):
        if (not self.action_space.contains(action)):
            return "ERROR THIS ACTION IS NOT PERMITTED"
        elif (self.state == self.terminal_state):# or (np.random.rand() < (1-self.success_rate)):
             return self.state, 0.0, self.done, None
        else:
            new_state = self.execute_action(action)
            reward = self.get_reward(new_state)
            self.state = new_state
            return self.state, reward, self.done, None
        
    def reset(self):
        self.state = 0 
        self.done = False
        return self.state
        
    
    def getStateFromXY(self, x, y):
        return x * self.grid_size + y
        
    def execute_action(self, action):   
        currX = self.state//self.grid_size
        currY = self.state % self.grid_size
        
        if(action==0): # Going Up
            if(np.random.rand() < self.prob_veerHorizontal):
                # Go Left
                newX = currX
                newY = max(0, currY - 1)
            elif(np.random.rand() < self.prob_veerHorizontal):
                # Go Right
                newX = currX
                newY = min(self.grid_size - 1, currY + 1)
            else:
                newX = max(0, currX-1)
                newY = currY
        elif(action==1): # Going Down
            if (np.random.rand() < self.prob_veerHorizontal):
                # Go Left
                newX = currX
                newY = max(0, currY - 1)
            elif (np.random.rand() < self.prob_veerHorizontal):
                # Go Right
                newX = currX
                newY = min(self.grid_size - 1, currY + 1)
            else:
                newX = min(self.grid_size-1, currX+1)
                newY = currY
        elif(action==2): # Going Left
            if (np.random.rand() < self.prob_veerHorizontal):
                # Go Dowm
                newX = min(self.grid_size - 1, currX + 1)
                newY = currY
            elif (np.random.rand() < self.prob_veerHorizontal):
                # Go Up
                newX = max(0, currX - 1)
                newY = currY
            else:
                newX = currX
                newY = max(0, currY-1)
        else:
            if (np.random.rand() < self.prob_veerHorizontal):
                # Go Dowm
                newX = min(self.grid_size - 1, currX + 1)
                newY = currY
            elif (np.random.rand() < self.prob_veerHorizontal):
                # Go Up
                newX = max(0, currX - 1)
                newY = currY
            else:
                newX = currX
                newY = min(self.grid_size-1, currY+1) # Going Right
            
        if(self.grid[newX][newY]==1): # We have hit an obstacle so the robot stays at its place
            return self.getStateFromXY(currX, currY)
        return self.getStateFromXY(newX, newY)
    
    def get_reward(self, next_state):
        x = next_state//self.grid_size
        y = next_state % self.grid_size
        if(self.waterMap[x][y]==1):
            return -10
        elif(next_state==self.terminal_state):
            self.done = True
            return 10
        else:
            return 0
           