import math

import numpy as np
from controller import Gyro, Robot, Supervisor
from Interfaces.BaseEnvironment import BaseEnvironment


class WebotsEnvironment(Supervisor,BaseEnvironment):
    """The robot's environment in Webots."""

    def __init__(self, writer=None):
        super().__init__()
        
        self.hit_count = 0
        self.reach_goal_count = 0
        self.current_step = 0
        self.writer = writer
        
        self.destination_coordinate = np.array([-3.13, 0])  # Target position
        self.reach_threshold = 0.1  # Distance threshold for reaching the goal
        self.floor_size = np.linalg.norm([8, 8])
        
        # Simulated position (instead of GPS)
        self.current_position = np.array([0.0, 0.0])  # Starting position

    def initialize_trajectory(self):
        return self.get_observations()
    
    def clear_grad(self):
        pass
    
    def reset_rewards(self):
        self.reach_goal_count = 0
        self.hit_count = 0
        
    def set_log_writer(self, writer):
        self.writer = writer
        
    def normalizer(self, value, min_value, max_value):
        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value

    def get_distance_to_goal(self):
        distance_to_goal = np.linalg.norm(self.destination_coordinate - self.current_position)
        normalized_distance = self.normalizer(distance_to_goal, min_value=0, max_value=self.floor_size)
        return normalized_distance

    def get_observations(self):
        # Simulated observation: just the normalized position and distance to goal
        norm_position = self.current_position / np.linalg.norm(self.current_position + 1e-8)
        distance_to_goal = self.get_distance_to_goal()
        state_vector = np.concatenate([norm_position, [distance_to_goal]], dtype=np.float32)
        return state_vector
        
    def reset(self):
        if self.writer is not None:
            self.writer.log({'hit_count': self.hit_count, 'goal_count': self.reach_goal_count})
        
        self.current_position = np.array([0.0, 0.0])  # Reset position
        self.current_step = 0
        return self.get_observations()

    def step(self, action, max_steps):
        # Simulate movement (instead of using motors)
        self.current_position += action  # Action is a 2D vector [dx, dy]
        
        step_reward, done = self.get_reward()
        state = self.get_observations()

        truncated = False
        if (self.current_step + 1) % max_steps == 0:
            truncated = True

        return state, step_reward, done, truncated

    def log_reward(self, name, score, return_dict=False):
        return score
                
    def get_reward(self):
        self.current_step += 1
        done = False
        reward = 0
        current_distance = self.get_distance_to_goal()

        # Continuous reward based on distance to goal
        reward += self.log_reward('distance', 1.4 / (current_distance + 1.0))

        if current_distance < 0.4:
            reward += self.log_reward('distance_04', 2.0 / (current_distance + 1.0))
        
        reach_threshold = self.reach_threshold * 100
        normalized_current_distance = current_distance * 100
            
        # Reward or punishment based on task completion
        # Simulated collision (e.g., if position goes out of bounds)
        if (np.abs(self.current_position) > self.floor_size).any():
            self.hit_count += 1
            done = True
            reward += self.log_reward('hit', -40)
        elif normalized_current_distance < reach_threshold:
            self.reach_goal_count += 1
            done = True
            reward += self.log_reward('goal', 75)
            print('+++ SOLVED +++')

        reward += self.log_reward('time_penalty', -0.02)
        return reward, done
