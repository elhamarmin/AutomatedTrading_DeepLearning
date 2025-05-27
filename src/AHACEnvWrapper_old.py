# matplotlib.use('Agg')
import datetime
import sys
from pprint import pprint


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import (backtest_plot, backtest_stats, get_baseline,
                        get_daily_return)
from stable_baselines3.common.logger import configure

from WebotsGymEnv import WebotsGymEnv

sys.path.append("../FinRL")

import itertools
import os

from finrl import config, config_tickers
from finrl.config import (DATA_SAVE_DIR, INDICATORS, RESULTS_DIR,
                          TENSORBOARD_LOG_DIR, TEST_END_DATE, TEST_START_DATE,
                          TRADE_END_DATE, TRADE_START_DATE, TRAIN_END_DATE,
                          TRAIN_START_DATE, TRAINED_MODEL_DIR)
from finrl.main import check_and_make_directories


class AHACEnvWrapper:
    def __init__(self,logdir=None,no_grad=None,num_envs=2048):
        super(AHACEnvWrapper, self).__init__()

        check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
        TRAIN_START_DATE = '2010-01-01'
        TRAIN_END_DATE = '2021-10-01'
        TRADE_START_DATE = '2021-10-01'
        TRADE_END_DATE = '2023-03-01'
        #Downloading data
        df = YahooDownloader(start_date = TRAIN_START_DATE,
                            end_date = TRADE_END_DATE,
                            ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
        
        #Preprocess
        fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = INDICATORS,
                    use_vix=True,
                    use_turbulence=True,
                    user_defined_feature = False)

        processed = fe.preprocess_data(df)
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date,list_ticker))

        processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date','tic'])

        processed_full = processed_full.fillna(0)
        mvo_df = processed_full.sort_values(['date','tic'],ignore_index=True)[['date','tic','close']]

        #Data split
        train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
        trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
        train_length = len(train)
        trade_length = len(trade)
        print(train_length)
        print(trade_length)
        stock_dimension = len(train.tic.unique())
        state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension

        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }

        env = StockTradingEnv(df = train, **env_kwargs)

        # Initialize the Webots environment
        self.env = env
        
        num_obs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.episode_length = 1000
    
    def to_tensor(self, np_arr):
        return torch.tensor(np_arr,dtype=torch.float32)
    
    def clear_grad(self):
        pass
    
    def initialize_trajectory(self):
        obs_data = self.env.state
        
        data = self.to_tensor(obs_data)
        
        return data
    
    def reset(self,seed: int | None = None):
        """Resets the environment to its initial state and returns the initial observation."""
        reset_data = self.env.reset()[0]
        
        data = self.to_tensor(reset_data)
        return data

    def step(self, action):
        """Takes a step in the environment based on the action."""
        
        _action = action.detach().numpy()
        state, reward, done,truncated,info = self.env.step(_action)
        
        reward = self.to_tensor(reward)
        state = self.to_tensor(state)
        acc_arr = torch.zeros(
                (1, 1,1),
                dtype=torch.float32,
                requires_grad=False,
            )
        cntc_arr = torch.zeros(
                (1, 1,1),
                dtype=torch.float32,
                requires_grad=False,
            )
        
        if done:
            termination = torch.ones(self.num_envs, dtype=torch.bool)
        else:
            termination = torch.zeros(self.num_envs, dtype=torch.bool)
            
        truncation = torch.zeros(self.num_envs, dtype=torch.bool)
        
        done = termination | truncation
        info = {
            "termination" : termination,
            "truncation" : truncation,
            "contact_forces" : cntc_arr,
            "accelerations" : acc_arr,
            "obs_before_reset" : state
        }
        return state, reward, done, info

    def render(self, mode='human'):
        """Renders the environment."""
        pass  # Implement render method if needed

    def close(self):
        """Performs any necessary cleanup."""
        pass  # Implement close method if needed
