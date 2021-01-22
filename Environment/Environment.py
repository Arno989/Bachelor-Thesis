#%%
import pandas as pd
import numpy as np
import random, os, datetime


class Environment:
    def __init__(self, data: pd.DataFrame(), window = 60):
        self.action_space = [0,1,2] # sell, hold, buy
        self.state_space = [window, 3]
        self.close_data = np.asarray(data["close"][1:])
        self.diff_data = np.asarray(data["close"][:-1]) - np.asarray(data["close"][1:])
        self.volume_data = np.asarray(data["volume"][1:])
        self.window = window
        
        self.reset()
        
    def reset(self):
        self.step = 0
        self.done = False
        self.balance = 0
        self.last_trade = []
        
        return self.observe()
    
    
    def step(self, action):
        
        if action == 0:
            pass
        elif action == 1: # sell
            pass
        elif action == 2: # buy
            pass
        else:
            raise ValueError(action)
        
        return self.observe()
    
    
    def observe(self):
        observation = [self.diff_data[self.step:self.step+60], self.volume_data[self.step:self.step+60]]
        
        return observation
    
    
    
# %%
df = pd.read_csv('../Data/Prices/AAPL.csv')
env = Environment(df)
print(env.diff_data)
# %%
