import numpy as np

class Random:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        
    def act(self, state):
        return np.random.choice(np.arange(self.action_space))