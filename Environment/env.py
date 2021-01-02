from subprocess import check_output

import time
import copy
import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L


class Environment1:
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history # obs
    
    def step(self, act):
        reward = 0
        
        # act = 0: stay, 1: buy, 2: sell
        if act == 1:
            self.positions.append(self.data.iloc[self.t, :]['close'])
        elif act == 2: # sell
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (self.data.iloc[self.t, :]['close'] - p)
                reward += profits
                self.profits += profits
                self.positions = []
        
        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['close'] - p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['close'] - self.data.iloc[(self.t-1), :]['close'])
        
        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        
        return [self.position_value] + self.history, reward, self.done # obs, reward, done



class Environment2:
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.cash = 10000
        self.purchase_budget = 500
        self.buys = []
        self.sell_value = 0
        self.short_buys = []
        self.short_sell_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.sell_value, self.short_sell_value] + self.history # obs
    
    def step(self, action):
        """
        one hot encoding 8 bits voor int
        reward = 10000
        
        0 -> stay
        
        1 -> buy -> -1 als onder 400 buy, -1% als boven 100 --> reward - buy value
        2 -> sell -> -1 als onder 400 buy, -1% als boven 100 --> reward + sell value
        
        3 -> short-buy -> -1 als onder 400 short-buy, -1% als boven 100 --> reward - buy value
        4 -> short-sell -> -1 als onder 400 short-buy, -1% als boven 100 --> reward + sell value
        """
        # reward function
        reward = 0
        fee = (self.purchase_budget/100)*1
        
        if action == 1: # buy
            reward -= self.purchase_budget
            reward -= fee
            self.buys.append(self.data.iloc[self.t, :]['close'])  # append buy price to buys list
        
        elif action == 2: # sell
            if len(self.buys) == 0: # penalty als verkoopt zonder stock
                reward = -10
            else:
                for p in self.buys:
                    reward += (self.purchase_budget / p) * self.data.iloc[self.t, :]['close']   # (budget / aankoopprijs) * verkoopprijs
                reward -= fee
                self.buys = []
        
        elif action == 3: # short-buy
            reward -= self.purchase_budget
            reward -= fee
            self.short_buys.append(self.data.iloc[self.t, :]['close'])  # append buy price to short_buys list
        
        elif action == 4: # short-sell
            if len(self.short_buys) == 0: # penalty als verkoopt zonder stock
                reward = -10
            else:
                for p in self.short_buys:
                    reward += ((self.purchase_budget / p) * self.data.iloc[self.t, :]['close'])*-1   # ((budget / aankoopprijs) * verkoopprijs) *-1
                reward -= fee
                self.short_buys = []
                
        
        # set next time
        self.t += 1
        self.sell_value = 0
        for p in self.buys:
            self.sell_value += (self.purchase_budget / p) * self.data.iloc[self.t, :]['close']
        self.short_sell_value = 0
        for p in self.short_buys:
            self.short_sell_value += ((self.purchase_budget / p) * self.data.iloc[self.t, :]['close'])*-1
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['close'] - self.data.iloc[(self.t-1), :]['close'])
        
        # clipping reward
        # if reward > 0:
        #     reward = 1
        # elif reward < 0:
        #     reward = -1
        
        return [self.sell_value, self.short_sell_value] + self.history, reward, self.done # obs, reward, done