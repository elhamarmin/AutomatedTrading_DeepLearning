# matplotlib.use('Agg')
import datetime
import sys
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

sys.path.append("../FinRL")
import itertools
import os
import re

import pandas as pd
import torch
import torch.nn as nn
from alpaca.data import StockBarsRequest, TimeFrame, TimeFrameUnit
from alpaca.data.historical import StockHistoricalDataClient
from finrl import config, config_tickers
from finrl.config import (DATA_SAVE_DIR, INDICATORS, RESULTS_DIR,
                          TENSORBOARD_LOG_DIR, TEST_END_DATE, TEST_START_DATE,
                          TRADE_END_DATE, TRADE_START_DATE, TRAIN_END_DATE,
                          TRAIN_START_DATE, TRAINED_MODEL_DIR)
from finrl.main import check_and_make_directories


def data_filler(df_dirty: pd.DataFrame):
    df = df_dirty.copy()

    # Ensure the date column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicate timestamps for each ticker
    df = df.drop_duplicates(subset=['tic', 'date'])
    # Get unique list of all dates
    unique_dates = pd.Series(df['date'].unique())

    # Get unique list of all ticks
    unique_tics = df['tic'].unique()

    # Create a complete set of tickers and dates
    complete_index = pd.MultiIndex.from_product([unique_tics, unique_dates], names=['tic', 'date'])

    # Reindex the original data to the complete index
    df_complete = df.set_index(['tic', 'date']).reindex(complete_index).sort_index()

    # Forward fill missing values
    df_filled = df_complete.groupby('tic').ffill()

    # Reset index to bring 'date' and 'tic' back as columns
    df_filled.reset_index(inplace=True)
    
    df_filled['date'] = df_filled['date'].astype(str)
    return df_filled

def alpaca_downloader(startdate, enddate,symbols, interval = "5m"):
    api_key = "Your_Key"
    api_secret = "Your_Secret"

    number = 5
    letter = "m"
    
    match = re.match(r"(\d+)([a-zA-Z]+)", interval)
    if match:
        number = int(match.group(1))
        letter = match.group(2)

    timeframeunit = TimeFrameUnit.Day
    
    if letter.lower() == 'd':
        timeframeunit = TimeFrameUnit.Day
    elif letter == 'm':
        timeframeunit = TimeFrameUnit.Minute
    elif letter == 'M':
        timeframeunit = TimeFrameUnit.Month
    elif letter.lower() == 'w':
        timeframeunit = TimeFrameUnit.Week
    elif letter.lower() == 'h':
        timeframeunit = TimeFrameUnit.Hour
        
    print(number,letter)
    timeframe = TimeFrame(number,timeframeunit)
    
    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    request_params = StockBarsRequest(symbol_or_symbols=symbols, start=startdate, end=enddate, timeframe=timeframe)
    bars = client.get_stock_bars(request_params=request_params)

    df = bars.df.reset_index()
    df.rename(columns={
        "timestamp":'date',
        "symbol":'tic'
    }, inplace=True)
    
    df = data_filler(df)
    df['date'] = df['date'].astype(str)
    
    return df

class AHACEnvWrapper:
    def __init__(self, logdir=None, no_grad=None, num_envs=2048):
        super(AHACEnvWrapper, self).__init__()
        
        self.discount = 0.99

        check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
        TRAIN_START_DATE = '2024-06-02'
        TRAIN_END_DATE = '2024-08-02'
        TRADE_START_DATE = '2024-04-01'
        TRADE_END_DATE = '2024-07-28'
        
        self.start_date = TRAIN_START_DATE
        self.end_date = TRAIN_END_DATE
        
        yahoo_data = False
        
        if yahoo_data:
            df = YahooDownloader(start_date=TRAIN_START_DATE,
                                end_date=TRAIN_END_DATE,
                                ticker_list=config_tickers.DOW_30_TICKER).fetch_data()
        else:
            interval = '5m'
            data_name = 'Dow30'
            save_path = f'./{TRAIN_START_DATE}_{TRAIN_END_DATE}_{data_name}_{interval}.csv'
            
            if os.path.exists(save_path):
                df = pd.read_csv(save_path).drop(columns=['Unnamed: 0'])
            else:
                df = alpaca_downloader(TRAIN_START_DATE,TRAIN_END_DATE,config_tickers.DOW_30_TICKER,interval=interval)
                df.to_csv(save_path)
            
        
        #Preprocess
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False)

        processed = fe.preprocess_data(df)
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date', 'tic'])

        processed_full = processed_full.fillna(0)
        mvo_df = processed_full.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]

        #Data split
        train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
        trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)


        print("Trian data index:")
        print(train.index.unique())
        
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
            "reward_scaling": 0.1,
            "print_verbosity": 1,
            'model_name':'ahac',
            'mode':'human',
            'make_plots':True
        }

        env = StockTradingEnv(df=train, **env_kwargs)

        # Initialize the Webots environment
        self.env = env
        
        self.num_envs = num_envs
        self.num_obs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.episode_length = 1000
        
        self.actions = []

    
    def to_tensor(self, np_arr):
        return torch.tensor(np_arr, dtype=torch.float32)
    
    def clear_grad(self):
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         param.grad.detach_()
        #         param.grad.zero_()
        pass
    
    def normalize(self, tensor):
        if tensor.numel() > 1 and tensor.std() != 0:
            return (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        return tensor

    def initialize_trajectory(self):
        
        self.clear_grad()
        
        obs_data = self.to_tensor(self.env.state)
        return obs_data
    
    def reset(self, seed=None):
        """Resets the environment to its initial state and returns the initial observation."""
        self.plot_actions()
        reset_data = self.env.reset()[0]
        return self.to_tensor(reset_data)

    def step(self, action):
        self.clear_grad()
        _action = action.detach().numpy()
        state, reward, done, truncated, info = self.env.step(_action)
        
        # print("State:", state)
        # print("Reward:", reward)
        # print("Done:", done)
        self.actions.append(_action)
    
        reward = self.to_tensor(reward)
        state = self.to_tensor(state)

        acc_arr = torch.zeros(
            (1, 1, 1),
            dtype=torch.float32,
            requires_grad=False,
        )
        cntc_arr = torch.zeros(
            (1, 1, 1),
            dtype=torch.float32,
            requires_grad=False,
        )
        
        termination = torch.ones(self.num_envs, dtype=torch.bool) if done else torch.zeros(self.num_envs, dtype=torch.bool)
        truncation = torch.zeros(self.num_envs, dtype=torch.bool)
        
        done = termination | truncation
        info = {
            "termination": termination,
            "truncation": truncation,
            "contact_forces": cntc_arr,
            "accelerations": acc_arr,
            "obs_before_reset": state,
        }
        
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset()
            
        return state, reward, done, info

    def render(self, mode='human'):
        """Renders the environment."""
        pass  # Implement render method if needed

    def close(self):
        """Performs any necessary cleanup."""
        pass  # Implement close method if needed


    def plot_actions(self):
        if not hasattr(self.env, 'asset_memory'):
            print("Environment does not have asset_memory.")
            return

        asset_memory = self.env.asset_memory

        date_range = self.env.date_memory

        if len(asset_memory) <= 1:
            return
        
        plot_data = pd.DataFrame({
            'date': date_range,
            'Assets': asset_memory
        })

        clean_timestamps = pd.to_datetime(plot_data['date']).dt.strftime('%m-%d %H:%M').array

        fig, ax = plt.subplots()
        fig.set_size_inches(8,6)

        plt.cla()
        plt.subplots_adjust(bottom=0.2)  # Adjust the bottom space as needed

        ax.plot(clean_timestamps,asset_memory, label='Assets', color='red')
        ax.set_xticks(clean_timestamps,labels=clean_timestamps,rotation=90)
        ax.set_xlabel('Date')
        ax.set_ylabel('Assets')
        ax.set_title('Assets over Time (5-Minute Intervals)')
        ax.grid(True)
        ax.legend()

        labels = ax.get_xticklabels()
        if len(labels) >= 40:
            for i, label in enumerate(labels):
                if i % 2 != 0:  # Show only even indexed labels
                    label.set_visible(False)
                
        plt.savefig(f'assets_plot_{self.env.episode}.png')  # Save the plot to a file
        plt.show()  # Display the plot
        
