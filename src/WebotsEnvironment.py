import math

import numpy as np
from controller import Gyro, Robot, Supervisor
from Interfaces.BaseEnvironment import BaseEnvironment


class WebotsEnvironment(Supervisor,BaseEnvironment):
    """The robot's environment in Webots."""

    def __init__(self, writer=None):
        robot = Robot()
        self.robot = robot
        
        super().__init__()
    
        
        self.hit_count = 0
        self.reach_goal_count = 0
        
        self.current_step = 0
        self.writer = writer
        # General environment parameters
        self.max_speed = 20  # Maximum Angular speed in rad/s
        self.max_rotation = 4  # Maximum Angular speed in rad/s
        
        self.destination_coordinate = np.array([-3.13, 0])  # Target (Goal) position
        self.reach_threshold = 0.1  # Distance threshold for considering the destination reached.
        self.safe_distance = 0.35
        self.floor_size = np.linalg.norm([8, 8])

        # Activate Devices
        self.bl_motors = []
        
        self.bl_motors.append(self.robot.getDevice('left_front_wheel_motor'))
        self.bl_motors.append(self.robot.getDevice('right_front_wheel_motor'))
        self.bl_motors.append(self.robot.getDevice('left_rear_wheel_motor'))
        self.bl_motors.append(self.robot.getDevice('right_rear_wheel_motor'))

        for motor in self.bl_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        self.servos = []
        
        self.servos.append(self.robot.getDevice('left_front_wheel_steering_motor'))
        self.servos.append(self.robot.getDevice('right_front_wheel_steering_motor'))
        
        
        # GPS Sensor
        sampling_period = 1  # in ms
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(sampling_period)
        
        # Gyro Sensor
        sampling_period = 1  # in ms
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(sampling_period)

        # Touch Sensor
        self.touch = self.robot.getDevice("touch sensor")
        self.touch.enable(sampling_period)
        
        self.compass = robot.getDevice("compass")
        self.compass.enable(sampling_period)
        
        self.lidar = robot.getDevice("lidar")
        self.lidar.enable(sampling_period)

        # List of all available sensors
        available_devices = list(self.robot.devices.keys())
        filtered_list = [item for item in available_devices if 'so' in item and any(char.isdigit() for char in item)]
        filtered_list = sorted(filtered_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        self.robot.step(200)  # take some dummy steps in environment for initialization

    
    def initialize_trajectory(self):
        return self.get_observations()
    
    def clear_grad(self):
        pass
    
    def reset_rewards(self):
        self.reach_goal_count = 0
        self.hit_count = 0
        
    def set_log_writer(self , writer):
        self.writer = writer
        
        
    def normalizer(self, value, min_value, max_value):
        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value

    def get_distance_to_goal(self):
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        distance_to_goal = np.linalg.norm(self.destination_coordinate - current_coordinate)
        
        normalizied_distance = self.normalizer(distance_to_goal, min_value=0, max_value=self.floor_size)
        
        return normalizied_distance
    
    def get_lidar_data(self):
        
        lidar = np.array(self.lidar.getRangeImage())
        lidar = self.normalizer(lidar , 0 , 1.5)
        lidar = np.clip(lidar , 0 , 1)
        
        return lidar
    
    def get_lidar_score(self):
        
        lidar_data = self.get_lidar_data()
        
        min_distance = np.min(lidar_data)
        if min_distance < self.safe_distance:
            return -5 * (self.safe_distance - min_distance)  # Penalty for being too close to walls
        else:
            return 1.6  # Reward for maintaining a safe distance

    def get_observations(self):

        lidar_data = self.get_lidar_data()
        
        gps_value = self.gps.getValues()[0:2]
        norm_gps_value = gps_value / np.linalg.norm(gps_value)
        vector_goal = self.get_vector_to_goal()
        
        compass_data = np.array(self.compass.getValues()[0:2])
        
        state_vector = np.concatenate([vector_goal,compass_data,norm_gps_value,lidar_data], dtype=np.float32)

        return state_vector
        
    def reset(self):
        if self.writer != None:
            self.writer.log({'hit_count' : self.hit_count , 'goal_count' : self.reach_goal_count})
        
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        
        return self.get_observations()

    def step(self, action, max_steps):
        self.apply_action(action)
        step_reward, done = self.get_reward()
        state = self.get_observations()  # New state

        truncated = False
        # Time-based termination condition
        if (int(self.getTime()) + 1) % max_steps == 0:
            truncated = True

        return state, step_reward, done, truncated

    def angle_between_vectors(self , v1, v2):
        # Compute the dot product
        dot_product = np.dot(v1, v2)
        
        # Compute the magnitudes (norms) of the vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # Compute the cosine of the angle
        cosine_angle = dot_product / (norm1 * norm2)
        
        # Ensure the cosine value is within the valid range for arccos
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Compute the angle in radians
        angle_radians = np.arccos(cosine_angle)
        angle_degree = math.degrees(angle_radians)
        
        return angle_degree

    def get_heading_vector(self):
        
        compass = np.array(self.compass.getValues()[0:2], dtype=np.float16)
        # Rotate the vector
        # rotated_vector = [-compass[1] , -compass[0]]
    
        return compass

    def get_vector_to_goal(self):
        gps_value = np.array(self.gps.getValues()[0:2], dtype=np.float32)

        x_r = gps_value[0]
        y_r = gps_value[1]
        
        x_g = self.destination_coordinate[0]
        y_g = self.destination_coordinate[1]
        
        # Calculate the direction to the goal
        vector_to_goal = np.array([x_g - x_r, y_g - y_r])
        vector_to_goal = vector_to_goal / np.linalg.norm(vector_to_goal)
        
        return vector_to_goal
    
    def look_at_score(self, threshold=10):
        """
        Calculate the reward or punishment for the robot based on its heading towards the goal.

        :param threshold: Angle threshold in degrees for determining if the robot is looking at the goal.
        :return: Reward (positive value) or punishment (negative value).
        """
        
        vector_to_goal = self.get_vector_to_goal()
        heading_vec = self.get_heading_vector()
        
        degree = self.angle_between_vectors(heading_vec , vector_to_goal)
        
        score = (threshold - degree) / threshold
        return score
    
    def get_gyro_score(self):
        
        abs_value = abs(self.gyro.getValues()[2])
        
        return -1 * abs_value

        
    def log_reward(self,name,score,return_dict = False):
        
        return score
    
                
    def get_reward(self):
        
        self.current_step += 1
        
        done = False
        reward = 0
        current_distance = self.get_distance_to_goal()

        # (1) Continuous reward based on distance to goal
        reward += self.log_reward('distance' , 1.4 / (current_distance + 1.0))

        if current_distance < 0.4:
            reward += self.log_reward('distance_04' , 2.0 / (current_distance + 1.0))
        
        reach_threshold = self.reach_threshold * 100
        normalized_current_distance = current_distance * 100
            
        # (3) Reward or punishment based on failure or completion of task
        check_collision = self.touch.value
        if normalized_current_distance < reach_threshold:
            self.reach_goal_count += 1
            
            done = True
            reward += self.log_reward('goal' ,75)
            print('+++ SOLVED +++')
        elif check_collision:
            self.hit_count += 1
            
            done = True
            reward += self.log_reward('hit' ,-40)
        
        # # (4) Penalty for proximity to obstacles
        lidar_score = self.get_lidar_score()
        
        if normalized_current_distance > 35:
            reward += self.log_reward('lidar_03' ,0.3 * lidar_score)
        else:
            reward += self.log_reward('lidar_1' , 1 * lidar_score)
        
        look_at_score_50 = self.look_at_score(50)
        if normalized_current_distance < 50:
            reward += self.log_reward('look_50' , 5.5 * look_at_score_50)
        
        look_at_score_60 = self.look_at_score(60)
        if normalized_current_distance < 20:
            reward += self.log_reward('look_60_dist' , 2.0 / (current_distance + 1.0))
            
            reward += self.log_reward('look_60' ,3.8 * look_at_score_60)
            
        reward += self.log_reward('time_penalty' , -0.02) 
        #reward -= 0.02

        
        return reward, done


    def apply_action(self, action):
        
        for motor in self.bl_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(self.max_speed * action[0])
            
        for motor in self.servos:
            motor.setPosition(action[1])

        self.robot.step(250)
        
        for motor in self.bl_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)
  