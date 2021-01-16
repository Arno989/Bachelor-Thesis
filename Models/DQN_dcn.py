from collections import deque
import os, random, csv
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Input
from tensorflow.keras.optimizers import Adam


class DQSN:
    def __init__(self, epsilon, gamma, epsilon_min, lr, epsilon_decay,  action_space, state_space, batch_size = 64, copy_interval = 1, sarsa = False):
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=50000)
        self.copy_interval = copy_interval
        self.fit_counter = 0
        self.action_space = action_space
        self.state_space = state_space
        self.model, self.target_model = self.build_model()
        self.sarsa = sarsa
    
    def build_model(self):
        _input = Input(shape = (60,1,))
        
        # Deconv segment, 1D convolution on 50 timesteps
        x = Conv1D(60, 2, activation='relu', padding='causal')(_input)
        x = Conv1D(40, 2, activation='relu', padding='causal')(x)
        x = Conv1D(20, 2, activation='relu', padding='causal')(x)
        # x = Conv1D(60, 2, activation='relu', padding='causal')(x)

        # FF part
        x = Dense(60, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(60, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(20, activation="relu")(x)
        x = Dense(self.action_space, activation="softmax")(x)

        _model = Model(_input, x)
        _model.compile(loss="mse", optimizer=Adam(lr=self.epsilon))
    
        return _model, _model
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.target_model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        if self.sarsa:
            targets = rewards + self.gamma * (self.model.predict_on_batch(next_states).shape[0]) * (1-dones)
        else:
            targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        self.fit_counter += 1
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.fit_counter % self.copy_interval == 0:
            self.target_model.set_weights(self.model.get_weights())
            # print("  Target network updated")
            
    def save_model(self):
        self.model.save("./Models/.h5/DQN-dcn.h5")

    def load_model(self):
        if os.path.isfile("./Models/.h5/DQN-dcn.h5"):
            self.model = load_model("./Models/.h5/DQN-dcn.h5")
            self.target_model = load_model("./Models/.h5/DQN-dcn.h5")



def train_dqsn_dcn(env, episodes, sarsa):
    hist_file = "./Data/Training Records/DQN_dcn.csv"
    
    gamma = .95
    learning_rate = 0.01
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = .995    
    
    ep_history = [] # reward, profit
    agent = DQSN(epsilon, gamma, epsilon_min, learning_rate, epsilon_decay, action_space=env.action_space.n, state_space=env.observation_space.shape[0], sarsa=sarsa)
    agent.load_model()
    
    for e in range(episodes):
        state = np.asarray([i[1] for i in env.reset()])
        # state = np.reshape(np.asarray([i[1] for i in env.reset()]),(64, 1, 60))
        print(state.shape)
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
        agent.save_model()
        
        try:
            open(hist_file, 'x')
        except:
            pass
        with open(hist_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(score)




