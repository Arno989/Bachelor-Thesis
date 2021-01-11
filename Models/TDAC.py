import os, random
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class TDAC:
    def __init__(self, alpha, beta, action_space, state_space, gamma=0.99):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.state_space = state_space
        self.n_actions = action_space

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]

    def build_actor_critic_network(self):
        # shared base network
        NN_input = Input(shape=(self.state_space,))
        delta = Input(shape=[1])
        dense1 = Dense(30, activation='relu')(NN_input)
        dense2 = Dense(30, activation='relu')(dense1)
        # final split layers
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        # actor loss function
        # def custom_loss(y_true, y_pred):
        #     out = K.clip(y_pred, 1e-8, 1-1e-8)
        #     log_lik = y_true * K.log(out)

        #     return K.sum(-log_lik * delta)
        
        def custom_loss(y_true, y_pred):
            action_true = y_true[:, :self.n_actions]
            advantage = y_true[:, self.n_actions:]

            return -tf.math.log(y_pred.prob(action_true) + 1e-5) * advantage
            # import tf.compat.v1 as tfc
            return -tfc.log(y_pred.prob(action_true) + 1e-5) * advantage

        # compose networks
        actor = Model(inputs=[NN_input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(inputs=[NN_input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(inputs=[NN_input], outputs=[probs])

        return actor, critic, policy

    def choose_action(self, state):
        # reshape for prediction
        state = state.reshape([1, state.shape[0]])
        # make probability distribution with predict 
        probabilities = self.policy.predict(state)[0]
        # return acion based on probabilities
        return np.random.choice(self.action_space, p=probabilities)

    def learn(self, state, action, reward, next_state, done):
        # reshape for prediction
        state = state.reshape([1, state.shape[0]])
        next_state = next_state.reshape([1, next_state.shape[0]])
        # predict critic values
        critic_value = self.critic.predict(state)
        critic_value_ = self.critic.predict(next_state)

        # calculate correction for action-value
        target = reward + self.gamma * critic_value_ * (1 - int(done))
        delta =  target - critic_value

        # no clue
        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        #update networks
        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0) 
        
    def save_model(self):
        self.actor.save("./Models/.h5/TDAC-actor.h5")
        self.critic.save("./Models/.h5/TDAC-critic.h5")
        self.policy.save("./Models/.h5/TDAC-policy.h5")

    def load_model(self):
        if os.path.isfile("./Models/.h5/TDAC-actor.h5"):
            self.actor = load_model("./Models/.h5/TDAC-actor.h5")
            self.critic = load_model("./Models/.h5/TDAC-critic.h5")
            self.policy = load_model("./Models/.h5/TDAC-policy.h5")