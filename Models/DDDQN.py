import os, csv

import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import load_model


# Network Class ---------------------------------------------------------------------------------------------
class DDDQN(tf.keras.Model):
    def __init__(self, action_space):
        super(DDDQN, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(action_space, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        Q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        x = self.d1(state)
        x = self.d2(x)
        a = self.a(x)
        return a
    
    
# Memory Class -------------------------------------------------------------------------------------------------
class exp_replay():
    def __init__(self, state_space, buffer_size= 1000000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, state_space), dtype=np.float32)
        self.action_mem = np.zeros((self.buffer_size), dtype=np.int32)
        self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, state_space), dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size), dtype=np.bool)
        self.pointer = 0

    def add_exp(self, state, action, reward, next_state, done):
        idx  = self.pointer % self.buffer_size 
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def sample_exp(self, batch_size= 64):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, next_states, dones


class Agent():
    def __init__(self, action_space, state_space, gamma=0.99, replace=100, lr=0.001):
        self.gamma = gamma
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 1e-3
        self.replace = replace
        self.trainstep = 0
        self.memory = exp_replay(state_space)
        self.batch_size = 64
        self.q_net = DDDQN(action_space=action_space)
        self.target_net = DDDQN(action_space=action_space)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)
        self.action_space = action_space
        self.state_space = state_space


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([i for i in range(self.action_space)])

        else:
            actions = self.q_net.advantage(np.array([state]))
            print(actions)
            action = np.argmax(actions)
            return action

    def update_mem(self, state, action, reward, next_state, done):
        self.memory.add_exp(state, action, reward, next_state, done)

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())     

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def train(self):
        if self.memory.pointer < self.batch_size:
            return 
        
        if self.trainstep % self.replace == 0:
            self.update_target()
        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_states)
        max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action]*dones
        self.q_net.train_on_batch(states, q_target)
        self.update_epsilon()
        self.trainstep += 1


    def save_model(self):
        self.q_net.save("./Models/.h5/DDDQN-Q_net.h5")
        self.target_net.save("./.h5/DDDQN-target.h5")

    def load_model(self):
        if os.path.isfile("./Models/.h5/DDDQN-Q_net.h5"):
            self.q_net = load_model("./Models/.h5/DDDQN-Q_net.h5")
            self.target_net = load_model("./Models/.h5/DDDQN-target.h5")



def train_dddqn(env, episodes):
    hist_file = "./Data/Training Records/DDDQN.csv"
    
    gamma = .99
    replace = 100
    lr = 0.001
    
    ep_history = []
    agent = Agent(action_space=env.action_space.n, state_space=env.observation_space.shape[0])

    for e in range(episodes):
        state = np.asarray([i[1] for i in env.reset()])
        done = False
        score = [0,0]
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.asarray([i[1] for i in next_state])
            
            agent.update_mem(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            
            score = [score[0] + reward, info["total_profit"]]

        ep_history.append(score)
        agent.save_model()
        
    try:
        open(hist_file, 'x')
    except:
        pass
    with open(hist_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for r in ep_history:
            writer.writerow(r)