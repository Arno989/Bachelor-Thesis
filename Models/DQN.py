from collections import deque
import os, random
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
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
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.sarsa = sarsa

    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_space, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model
    
    # def build_model(self):
    #     _input = Input(shape = 60, 1))
        
    #     # Deconv segment, 1D convolution on 50 timesteps
    #     x = Conv1D(60, 2, activation='relu', padding='causal')(_input)
    #     x = Conv1D(40, 2, activation='relu', padding='causal')(x)
    #     x = Conv1D(20, 2, activation='relu', padding='causal')(x)
    #     # x = Conv1D(60, 2, activation='relu', padding='causal')(x)

    #     # FF part
    #     x = Dense(60, activation="relu")(x)
    #     x = Dropout(0.5)(x)
    #     x = Dense(60, activation="relu")(x)
    #     x = Dropout(0.25)(x)
    #     x = Dense(self.action_space, activation="softmax")(x)

    #     _model = Model(_input, x)
    #     _model.compile(loss="mse", optimizer=Adam(lr=self.epsilon))

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
        self.model.save("./Models/.h5/DQN.h5")

    def load_model(self):
        if os.path.isfile("./Models/.h5/DQN.h5"):
            self.model = load_model("./Models/.h5/DQN.h5")
            self.target_model.set_weights(self.model.get_weights())




















"""
class DQSN:
    def __init__(self, epsilon, gamma, epsilon_min, lr, epsilon_decay, action_space, state_space, batch_size = 64, copy_interval = 1, sarsa = False, memory = 5000):
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory)
        self.copy_interval = copy_interval
        self.fit_counter = 0
        self.action_space = action_space
        self.state_space = state_space
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.sarsa = sarsa
        if self.sarsa:
            self.model_path = f'./Models/dsn.h5'
        else:
            self.model_path = f'./Models/dqn.h5'

    def build_model(self):
        _input = Input(shape = (60,2,))
        
        # Deconv segment, 1D convolution on 50 timesteps
        x = Conv1D(120, 2, activation='relu', padding='causal')(_input)
        x = Conv1D(80, 2, activation='relu', padding='causal')(x)
        x = Conv1D(60, 2, activation='relu', padding='causal')(x)
        x = Conv1D(40, 2, activation='relu', padding='causal')(x)

        # FF part
        x = Dense(120, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(120, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(self.action_space, activation="softmax")(x)

        _model = Model(_input, x)
        _model.compile(loss="mse", optimizer=Adam(lr=self.epsilon))
        print(_model.summary())
        return _model
    
        # model = Sequential()
        # model.add(Input(shape=self.state_space))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(units=self.action_space, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        # print(model.summary())
        # return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        print(state)
        print(state.shape) #shape 1,120 van maken
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
            targets = rewards + self.gamma * (np.argmax(np.amax(self.target_model.predict_on_batch(next_states), axis=1), axis=1))*(1-dones)
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
        self.model.save(self.model_path)
        #print('model saved')
        
    def load_model(self):
        if os.path.isfile(self.model_path):            
            self.model.load_weights(self.model_path)
            self.target_model.load_weights(self.model_path)
            print('model loaded')
"""
