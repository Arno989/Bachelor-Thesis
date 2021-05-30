import csv, time, math
import numpy as np
from dataloader import env_initialiser

class Random:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        
    def act(self, state):
        return np.random.choice(np.arange(self.action_space))


def train_random(episodes):
    env = env_initialiser().init()
    hist_file = "./Data/Training Records/Random.csv"
    ep_history = [] # [reward, profit, (profit/max_profit)*100, max_profit]
    agent = Random(action_space=env.action_space.n, state_space=env.observation_space.shape[0])
    
    run_start = time.time()
    timings = []
    
    for e in range(episodes):
        ep_start_time = time.time()
        
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
        
        timings.append(time.time()-ep_start_time)
        avg_time = sum(timings)/len(timings)
        m, s = divmod(math.floor(avg_time*(episodes-e)), 60)
        h, m = divmod(m, 60)
        
        print(f'\rEpisode: {e}/{episodes}, Time estimate: {math.floor(time.time() - run_start)}s/{math.floor(avg_time*500)}s => {h:d}:{m:02d}:{s:02d}.', end='', flush=True)