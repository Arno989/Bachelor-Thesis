import os, random
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout
from tensorflow.keras.optimizers import Adam


class AC:
    def __init__(self, alpha, beta, action_space, state_space, gamma=0.99, layer1_size=1024, layer2_size=512):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = state_space
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = action_space

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]

    def build_actor_critic_network(self):
        NN_input = Input(shape=(self.input_dims[0],))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(NN_input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)

        actor = Model(inputs=[NN_input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(inputs=[NN_input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(inputs=[NN_input], outputs=[probs])

        return actor, critic, policy

    def choose_action(self, state):
        state = state.reshape([1, state.shape[0]])
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta =  target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0) 