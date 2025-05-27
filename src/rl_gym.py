import datetime

from controller import Robot
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import HParam
from wandb.integration.sb3 import WandbCallback

import wandb
from WebotsEnvironment import WebotsEnvironment
from WebotsGymEnv import WebotsGymEnv

from shac.algorithms.ahac import AHAC

env = WebotsGymEnv()
check_env(env)

def manual_train_model(learning_rate,ent_coef,gae_lambda,n_steps,batch_size,n_epochs,clip_range,total_timesteps):
    
    ahac = AHAC()
    env.env.reset_rewards()
    
    model = PPO('MlpPolicy', env,learning_rate=learning_rate, 
                ent_coef=ent_coef,
                gae_lambda=gae_lambda,tensorboard_log='./gym_logs/',
                n_steps=n_steps,batch_size=batch_size,n_epochs=n_epochs, clip_range=clip_range, verbose=1)
    
    model = model.learn(total_timesteps=total_timesteps)
    
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    model.save(f'./trained_model/model_{time}.zip')
    
    return model


def train_model():
    run = wandb.init(
        project="sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    
    env.env.reset_rewards()
    env.env.set_log_writer(run)
    
    config = wandb.config
    
    model = PPO('MlpPolicy', env,learning_rate=config.learning_rate, 
                ent_coef=config.ent_coef,
                gae_lambda=config.gae_lambda,tensorboard_log='./gym_logs/',
                n_steps=config.n_steps,batch_size=config.batch_size,n_epochs=config.n_epochs, clip_range=config.clip_range, verbose=1)
    
    model = model.learn(total_timesteps=config.total_timesteps,
                        callback=WandbCallback(
                            gradient_save_freq=100,
                            model_save_path=f"models/{run.id}",
                            verbose=2,
                        ))
    
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    model.save(f'./trained_model/model_{run.id}_{time}.zip')
    
    wandb.finish()
    
    return model
    
run_model = 0 # 0 = Trian , 1 = Test , 2 = Debug

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "goal_count"},
    "parameters": {
        "gae_lambda": {"max": 0.85, "min": 0.75},
        "ent_coef": {'values': [0.1, 0.12, 0.15 ,0.17,0.2,0.24,0.26,0.3]},
        "clip_range" : {"max": 0.15, "min": 0.05},
        "total_timesteps": {'value': 30000},
        'learning_rate': {
            'values': [0.0001, 0.0002, 0.0003 ,0.0004]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'n_epochs':{
            'values': [64, 128, 256]
        },
        'n_steps': {
            'values': [1024, 512, 256]
        }
    },
}

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="sb3")
    
if run_model == 0:
    manual_train_model(learning_rate=0.0003,total_timesteps=30000,ent_coef=0.15,clip_range=0.1478,gae_lambda=0.7755,
                        n_epochs=64,n_steps=1024,batch_size=64)
    # wandb.agent(sweep_id,project='sb3' ,function=train_model, count=30)
    
# elif run_model == 1:
     # model.load('./trained_model/model_20240613-1747.zip')
elif run_model == 2:
    env = WebotsEnvironment(robot)
    

obs = env.reset()[0]
env.reset()

print("Finished")

while True:
    if run_model == 0:
        break
    
    if run_model == 1:
        val_obs = obs
        
        action, _states = model.predict(val_obs)
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
    
    # env.look_at_score(60)
    
    # vector_to_goal = env.get_vector_to_goal()
    # heading_vec = env.get_heading_vector()
    
    # compass = np.array(env.compass.getValues()[0:2], dtype=np.float16)
    
    # degree = env.angle_between_vectors(heading_vec , vector_to_goal)
    
    # print(f'goal: {vector_to_goal}')
    # print(f'head: {heading_vec}')
    # print(f'compass: {compass}')
    # print(f'degree: {degree}')
    
    if run_model == 2:
        key = input("Enter Key:")
    
        if key == 'a':
            env.apply_action([0,-1])
        elif key == 'd':
            env.apply_action([0,1])
        elif key == 'w':
            env.apply_action([1,0])
        elif key == 's':
            env.apply_action([-1,0])
        elif key == 'i':
            env.robot.step(500)
            
        lidar_data = env.get_lidar_score()
        print(lidar_data)
        
    
