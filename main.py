#%%
import gym, gym_anytrading
import os, time, random, datetime, csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from Models.Random import train_random
from Models.DQN import train_dqsn
from Models.PG import train_pg
from Models.TDAC import train_ac
from Models.DDDQN import train_dddqn
from Models.DQN_dcn import train_dqsn_dcn
from Models.DQN_lstm import train_dqsn_lstm

# Configure GPU memory growth and eager execution for agents
try:
    tf.compat.v1.disable_eager_execution()
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    print("GPU Configured")
except Exception as e:
    print("No GPU's detected\n",e)


### Environment setup -------------------------------------------------------------------------------------------------

# Slice and prepare data for environment
columnNames = ["Symbol", "Timestamp", "Open", "High", "Low", "Close", "Volume"]
data_len = 7 # slice of data in in days (1 min interval) (1 week is 2100 datapoints)

df = pd.read_csv('./Data/Prices/AAPL.csv')
df.columns = columnNames
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df[(df["Timestamp"] >= df["Timestamp"].max()-datetime.timedelta(days=data_len)) & (df["Timestamp"] <= df["Timestamp"].max())]

# step size = 60 minutes (make prediction every hour)
window_size = 60
start_index = window_size
end_index = len(df)

env = gym.make('stocks-v0', df = df, window_size = window_size, frame_bound = (start_index, end_index))
print(f"Actions: {env.action_space.n}, Observation space: {env.observation_space.shape[0]}")


#%%

episodes = 500

# train_random(env, episodes)
# train_pg(env, episodes)
# train_ac(env, episodes)
# train_dqsn(env, episodes, sarsa = False)
train_dqsn_dcn(env, episodes, sarsa = False)
# train_dqsn_lstm(env, episodes, sarsa = False)


# %%
