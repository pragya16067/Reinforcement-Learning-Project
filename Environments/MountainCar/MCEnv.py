# import all dependencies
import numpy as np
import gym
from gym import error, spaces, utils
from MountainCarBase import MountainCarEnv
import copy
from gym.utils import seeding

class MCEnv(MountainCarEnv):
    
    def __init__(self, pos_partitions=6, vel_partitions=2):
        
        self.pos_partitions = pos_partitions
        self.vel_partitions = vel_partitions
        self.n_states = (self.observation_space.high - self.observation_space.low)*\
                    np.array([pos_partitions, vel_partitions])
        self.n_states = np.round(self.n_states, 0).astype(int) + 1
        
        # Define the Observation Space
        self.terminal_state = self.n_states - 1
             
        
    def step(self, action):
        # Discretize state
        new_state, reward, done, info = self.step(action)
        state_adj = (new_state - self.observation_space.low)*np.array([self.pos_partitions, self.vel_partitions])
        state_adj = np.round(state_adj, 0).astype(int)
        
        return state_adj, reward, done, info
        
    def reset(self):
        # Discretize state
        state = self.reset()
        state_adj = (state - self.observation_space.low)*np.array([self.pos_partitions, self.vel_partitions])
        state_adj = np.round(state_adj, 0).astype(int)
        
        return state_adj
        