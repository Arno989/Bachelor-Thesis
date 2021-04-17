import csv
import numpy as np
import matplotlib.pyplot as plt
from dataloader import env_initialiser

class Random:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        
    def act(self, state):
        return np.random.choice(np.arange(self.action_space))

## TODO
# make per episode env reset
# change data to relative score against max profit
# randomise start date in data

def train_random(episodes):
    env = env_initialiser().init()
    hist_file = "./Data/Training Records/Random.csv"
    ep_history = [] # [reward, profit, (profit/max_profit)*100, max_profit]
    agent = Random(action_space=env.action_space.n, state_space=env.observation_space.shape[0])
    
    for e in range(episodes):
        env = env_initialiser().init()
        max_profit = env.max_possible_profit()
        state = np.asarray([i[1] for i in env.reset()])
        done = False
        score = [0,0]
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.asarray([i[1] for i in next_state])
            
            state = next_state
            
            score = [score[0] + reward, info["total_profit"]]
            
        score.append((score[1]/max_profit)*100)
        score.append(env.max_possible_profit())
        ep_history.append(score)
        
        try:
            open(hist_file, 'x')
        except:
            pass
        with open(hist_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(score)