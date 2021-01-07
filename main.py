#%%
import gym, gym_anytrading
import os, time, random, datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from Models.DQN import DQSN
from Models.PG import PG

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)



### Environment setup -------------------------------------------------------------------------------------------------

columnNames = ["Symbol", "Timestamp", "Open", "High", "Low", "Close", "Volume"]
data_len = 7 # slice of data in in days (1 min interval) (1 week is 2100 datapoints)

df = pd.read_csv('./Data/Prices/AAPL.csv')
df.columns = columnNames
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df[(df["Timestamp"] >= df["Timestamp"].max()-datetime.timedelta(days=data_len)) & (df["Timestamp"] <= df["Timestamp"].max())]

window_size = 60
start_index = window_size
end_index = len(df)

env = gym.make('stocks-v0', df = df, window_size = window_size, frame_bound = (start_index, end_index))
print(f"Actions: {env.action_space.n}, Observation space: {env.observation_space.shape[0]*env.observation_space.shape[1]}")



### Agent training functions --------------------------------------------------------------------------------------------

def train_dqsn(episodes, sarsa, render):
    rewardlist = []
    agent = DQSN(epsilon, gamma, epsilon_min, learning_rate, epsilon_decay, action_space=env.action_space.n, state_space=env.observation_space.shape[0], sarsa=sarsa)
    
    for e in range(episodes):
        state = np.asarray([i[1] for i in env.reset()])
        done = False
        score = 0
        i = 0
        
        while not done:
            action = agent.act(state)
            if render:
                env.render()
            next_state, reward, done, _ = env.step(action)
            next_state = np.asarray([i[1] for i in next_state])
                        
            score += reward
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            i += 1
                        
        rewardlist.append(score)
        
    env.render_all()
    return rewardlist


def train_pg(episodes, render):
    rewards = []
    cum_rewards = []
    
    agent = PG(gamma, lr_ml, lr_dl, env.action_space.n, (60, 1)) #env.observation_space.shape
    
    for e in range(episodes):
        # drop price, keep price change
        state = np.asarray([i[1] for i in env.reset()]) # np.reshape( ),(-1,1))
        score = 0
        done = False
        
        while not done:
            # play an acion and record the game state & reward per episode
            action, prob = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.asarray([i[1] for i in next_state])
            agent.remember(state, action, prob, reward)
            state = next_state
            
            score += reward
            
            if done:
                history = agent.train_policy_network()
                        
        rewards.append(score)
        cum_rewards.append(sum(rewards))
         
    env.render_all()
    return rewards, cum_rewards


# %%
if __name__ == '__main__':
    gamma = .95
    learning_rate = 0.01
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = .995
    SARSA = False
    render = False
    
    try:
        rewardlist = train_dqsn(10, SARSA, render)
    except KeyboardInterrupt as e:
        env.close()

# %%
if __name__ == '__main__':
    gamma = .90
    lr_ml = 0.01
    lr_dl = 0.01
    render = False
    render_interval = 1
    
    try:
        rewards, cum_rewards = train_pg(5, render)
    except KeyboardInterrupt as e:
        env.close()


# %%
