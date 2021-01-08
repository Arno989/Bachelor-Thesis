#%%
import gym, gym_anytrading
import os, time, random, datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from Models.Random import Random
from Models.DQN import DQSN
from Models.PG import PG
from Models.TDAC import TDAC
from Models.DDDQN import Agent


try:
    tf.python.framework.ops.disable_eager_execution()
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    print("GPU Configured")
except:
    print("No GPU's detected")



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



### Agent training functions -----------------------------------------------------------------------------------------------

def train_random(episodes):
    ep_history = [] # reward, profit
    agent = Random(action_space=env.action_space.n, state_space=env.observation_space.shape[0])
    
    for e in range(episodes):
        state = np.asarray([i[1] for i in env.reset()])
        done = False
        score = [0,0]
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.asarray([i[1] for i in next_state])
            
            state = next_state
            
            score = [score[0] + reward, info["total_profit"]]
            
                        
        ep_history.append(score)
        
    env.render_all()
    return ep_history


def train_dqsn(episodes, sarsa):
    gamma = .95
    learning_rate = 0.01
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = .995    
    
    ep_history = [] # reward, profit
    agent = DQSN(epsilon, gamma, epsilon_min, learning_rate, epsilon_decay, action_space=env.action_space.n, state_space=env.observation_space.shape[0], sarsa=sarsa)
    
    for e in range(episodes):
        state = np.asarray([i[1] for i in env.reset()])
        done = False
        score = [0,0]
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.asarray([i[1] for i in next_state])
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            
            score = [score[0] + reward, info["total_profit"]]
            
                        
        ep_history.append(score)
        
    env.render_all()
    return ep_history


def train_pg(episodes):
    gamma = .90
    lr_ml = 0.01
    lr_dl = 0.01
    
    ep_history = []
    agent = PG(gamma, lr_ml, lr_dl, env.action_space.n, env.observation_space.shape[0])
    
    for e in range(episodes):
        state = np.asarray([i[1] for i in env.reset()])
        done = False
        score = [0,0]
        
        while not done:
            action, prob = agent.compute_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.asarray([i[1] for i in next_state])
            
            agent.remember(state, action, prob, reward)
            state = next_state
            if done:
                agent.train_policy_network()
            
            score = [score[0] + reward, info["total_profit"]]
                        
        ep_history.append(score)
         
    env.render_all()
    return ep_history


def train_ac(episodes):
    alpha = 0.00001
    beta = 0.00005
    
    ep_history = []
    agent = TDAC(alpha, beta, action_space= env.action_space.n, state_space=env.observation_space.shape[0])

    for e in range(episodes):
        state = np.asarray([i[1] for i in env.reset()])
        done = False
        score = [0,0]
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.asarray([i[1] for i in next_state])
            
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            
            score = [score[0] + reward, info["total_profit"]]

        ep_history.append(score)
    
    env.render_all()
    return ep_history


def train_dddqn(episodes):
    gamma = .99
    replace = 100
    lr = 0.001
    
    ep_history = []
    agent = Agent(action_space=env.action_space.n, state_space=env.observation_space.shape[0])

    for e in range(episodes):
        state = np.asarray([i[1] for i in env.reset()])
        done = False
        score = [0,0]
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.asarray([i[1] for i in next_state])
            
            agent.update_mem(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            
            score = [score[0] + reward, info["total_profit"]]

        ep_history.append(score)
    
    env.render_all()
    return ep_history



# %%
def execute(episodes):
    Random_history = train_random(episodes)
    # DQN_history = train_dqsn(episodes, sarsa = False)
    # PG_history = train_pg(episodes)
    # TDAC_history = train_ac(episodes)
    # DDDQN_history = train_dddqn(episodes)
    
    return Random_history    
    # return [Random_history, DQN_history, PG_history, TDAC_history, DDDQN_history]


print(execute(1))


# %%
