import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout
from tensorflow.keras.optimizers import Adam

class critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation = None)

    def call(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v
    

class actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128,activation='relu')
        self.a = tf.keras.layers.Dense(2,activation='softmax')

    def call(self, input_data):
        x = self.d1(input_data)
        a = self.a(x)
        return a
 
 
class Agent0():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=7e-3)
        self.actor_net = actor()
        self.critic_net = critic()

          
    def act(self,state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tf.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])


class Agent():
    def __init__(self, env_actions, env_states, gamma=0.99, epsilon=0.9):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = env_actions
        self.state_space = env_states
        self.critic = critic()
        self.actor = actor()
        
        # record observations
        self.states = []
        self.gradients = [] 
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        
    
    # Init
    def critic(self):
        _input = Input(shape=self.state_space[1], batch_size=self.state_space[0])
        _ = Dropout(0.3)(_input)
        _ = Dense(20, activation="relu")(_)
        _ = Dropout(0.3)(_)
        _ = Dense(20, activation="relu")(_)
        _ = Dense(1, activation="relu")(_)
        
        _model = Model(_input, _)
        _model.compile(loss="mse", optimizer=Adam(lr=self.epsilon))
        print("Critic Network\n", _model.summary())
        return _model
    
    def actor(self):
        _input = Input(shape=self.state_space[1], batch_size=self.state_space[0])
        _ = Dropout(0.3)(_input)
        _ = Dense(20, activation="relu")(_)
        _ = Dropout(0.3)(_)
        _ = Dense(20, activation="relu")(_)
        _ = Dense(self.action_space, activation="softmax")(_)
        
        _model = Model(_input, _)
        _model.compile(loss="mse", optimizer=Adam(lr=self.epsilon))
        print("Network\n", _model.summary())
        return _model
    
    
    # Utilities
    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_shape)
        action_encoded[action] = 1

        return action_encoded

    def remember(self, state, action, action_prob, reward):
        encoded_action = self.hot_encode_action(action)
        self.gradients.append(encoded_action - action_prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)

    def discount_rewards(self, rewards):
        discounted_rewards=[]
        cumulative_total_return=0
        
        # iterate the rewards backwards and and calc the total return 
        for reward in rewards[::-1]:            
            cumulative_total_return=(cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        normalized_discounted_rewards = (discounted_rewards - mean_rewards) / (std_rewards + 1e-7) # avoiding zero div
        
        return normalized_discounted_rewards
        
        
    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        # get action probabilities
        action_probabilities = self.actor_net.predict(state).flatten()
        # normalise action probabilities
        action_probabilities /= np.sum(action_probabilities)
        #sample action
        action = np.random.choice(self.action_space, 1, p=action_probabilities)[0]
        
        return action, action_probabilities
        # state_values = self.actor_net.predict(state)
        
        
    def train_policy_network(self):
        # get X_train
        states = np.vstack(self.states)

        # get y_train
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        gradients *= self.get_discounted_rewards(rewards)
        gradients = self.alpha * np.vstack([gradients]) + self.probs
        
        history = self.model.train_on_batch(states, gradients)
        
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
        
        return history
    