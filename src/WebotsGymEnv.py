import gymnasium as gym
import numpy as np
from WebotsEnvironment import WebotsEnvironment
from controller import Robot
from gymnasium import spaces

class WebotsGymEnv(gym.Env):
    def __init__(self,logdir=None):
        super(WebotsGymEnv, self).__init__()
        
        # Initialize the Webots environment
        self.env = WebotsEnvironment()
        self.max_step = 1000
        
        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1,-1]),
                               high=np.array([1,1]),
                               dtype=np.float32)  # [0: forward, 1: right/left]
        
        num_obs = len(self.env.get_observations())
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_obs,), dtype=np.float32)
    
    def reset(self,seed: int | None = None):
        """Resets the environment to its initial state and returns the initial observation."""
        return (self.env.reset(),{})

    def step(self, action):
        """Takes a step in the environment based on the action."""
        state, reward, done,truncated = self.env.step(action, self.max_step)
        return state, reward, done, truncated, {}

    def render(self, mode='human'):
        """Renders the environment."""
        pass  # Implement render method if needed

    def close(self):
        """Performs any necessary cleanup."""
        pass  # Implement close method if needed
