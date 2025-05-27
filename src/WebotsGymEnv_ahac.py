import gymnasium as gym
import numpy as np
import torch
from WebotsEnvironment import WebotsEnvironment
from controller import Robot
from gymnasium import spaces

class WebotsGymEnv_ahac:
    def __init__(self,logdir=None,no_grad=None):
        super(WebotsGymEnv_ahac, self).__init__()
        
        # Initialize the Webots environment
        self.env = WebotsEnvironment()

        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1,-1]),
                               high=np.array([1,1]),
                               dtype=np.float32)  # [0: forward, 1: right/left]
        
        num_obs = len(self.env.get_observations())
        num_actions = self.action_space.shape[0]
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_obs,), dtype=np.float32)
        self.observation_space.shape[0]
        
        self.num_envs = 1
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.episode_length = 1000
    
    def to_tensor(self, np_arr):
        return torch.tensor(np_arr,dtype=torch.float32)
    
    def clear_grad(self):
        pass
    
    def initialize_trajectory(self):
        return self.reset()
    
    def reset(self,seed: int | None = None):
        """Resets the environment to its initial state and returns the initial observation."""
        data = self.to_tensor(self.env.reset())
        return data

    def step(self, action):
        """Takes a step in the environment based on the action."""
        state, reward, done,truncated = self.env.step(action, self.episode_length)
        
        reward = self.to_tensor(reward)
        
        state = self.to_tensor(state)
        info = {
            "termination" : truncated,
            "truncation" : truncated,
            "contact_forces" : 0,
            "accelerations" : 0,
            "obs_before_reset" : state
        }
        return state, reward, done, info

    def render(self, mode='human'):
        """Renders the environment."""
        pass  # Implement render method if needed

    def close(self):
        """Performs any necessary cleanup."""
        pass  # Implement close method if needed
