#%%
import gym, gym_anytrading
import os, time, random, datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from Models.DQN import DQSN

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs:", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)





# %%
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
print(env.observation_space.shape)
print(env.action_space)




# %%
def train_dqsn(episodes, sarsa, render):
    rewardlist = []
    agent = DQSN(epsilon, gamma, epsilon_min, learning_rate, epsilon_decay, action_space=env.action_space.n, state_space=env.observation_space.shape[0] * env.observation_space.shape[1], sarsa=sarsa)
    agent.load_model()
    
    for e in range(episodes):
        state = env.reset()
        done = False
        #state = state.flatten()                    # reshape naar aantal states variabelen
        score = 0
        i = 0
        
        while not done:
            action = agent.act(state)
            if render:
                env.render()
            next_state, reward, done, _ = env.step(action)
            #next_state = next_state.flatten()       # reshape naar aantal states variabelen
                        
            score += reward
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            i += 1
            
        if e % 10 == 0:
            agent.save_model()
            
        rewardlist.append(score)
        
    return rewardlist




#%%
if __name__ == '__main__':
    gamma = .95
    learning_rate = 0.01
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = .995
    SARSA = False
    render = False
    
    try:
        rewardlist = train_dqsn(1, SARSA, render)
    except KeyboardInterrupt as e:
        env.close()
# %%
