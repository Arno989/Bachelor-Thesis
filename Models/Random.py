import csv
import numpy as np
import matplotlib.pyplot as plt

class Random:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        
    def act(self, state):
        return np.random.choice(np.arange(self.action_space))



def train_random(env, episodes):
    hist_file = "./Data/Training Records/Random.csv"
    ep_history = [] # [reward, profit]
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
        
    try:
        open(hist_file, 'x')
    except:
        pass
    with open(hist_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for r in ep_history:
            writer.writerow(r)