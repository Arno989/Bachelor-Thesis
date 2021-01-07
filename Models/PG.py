import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input


class PG:
    def __init__(self, gamma, lr_ml, lr_dl, env_actions, env_states):
        self.action_space = env_actions
        self.state_space = env_states
        self.gamma = gamma
        self.alpha = lr_ml # ML
        self.learning_rate = lr_dl # DL
        self.model = self.build_policy_network()

        # record observations
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        
        
    def build_policy_network(self):
        _input = Input(shape=self.state_space[0])
        _ = Dropout(0.3)(_input)
        _ = Dense(20, activation="relu")(_)
        _ = Dropout(0.3)(_)
        _ = Dense(20, activation="relu")(_)
        _ = Dense(self.action_space, activation="softmax")(_)
        
        _model = Model(_input, _)
        _model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return _model

    
    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_space)
        action_encoded[action] = 1

        return action_encoded


    def remember(self, state, action, action_prob, reward):
        encoded_action = self.hot_encode_action(action)
        self.gradients.append(encoded_action - action_prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)


    def compute_action(self, state):  # COMPUTE THE ACTION FROM THE SOFTMAX PROBABILITIES
        # transfer rate
        state = state.reshape([1, state.shape[0]])
        # get actian probably
        action_probability_distribution = self.model.predict(state).flatten()
        # norm action probability distribution
        action_probability_distribution /= np.sum(action_probability_distribution)

        # sample action
        action= np.random.choice(self.action_space, 1, p = action_probability_distribution)[0]

        return action, action_probability_distribution


    def get_discounted_rewards(self, rewards):
        discounted_rewards=[]
        cumulative_total_return=0
        
        # iterate the rewards backwards and and calc the total return 
        for reward in rewards[::-1]:            
            cumulative_total_return=(cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards=np.mean(discounted_rewards)
        std_rewards=np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards - mean_rewards) / (std_rewards + 1e-7) # avoiding zero div
        
        return norm_discounted_rewards

    
    def train_policy_network(self):
        # get X_train
        states = np.vstack(self.states)

        # get y_train
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        discounted_rewards = self.get_discounted_rewards(rewards)
        gradients *= discounted_rewards
        y_train = self.alpha * np.vstack([gradients]) + self.probs
        
        # y_train = np.reshape(gradients, (gradients.shape[0], -1, gradients.shape[1]))
        # states = np.reshape(states, (states.shape[0], -1, states.shape[1]))
        
        history = self.model.train_on_batch(states, y_train)
        
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

        return history