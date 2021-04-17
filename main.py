#%%
import time
import tensorflow as tf

from Models.Random import train_random
from Models.DQN import train_dqsn
from Models.PG import train_pg
from Models.TDAC import train_ac
from Models.DQN_lstm import train_dqsn_lstm

# Configure GPU memory growth and eager execution for agents
try:
    tf.compat.v1.disable_eager_execution()
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    print("GPU Configured")
except Exception as e:
    print("No GPU's detected\n",e)


#%%
episodes = 500

start = time.time()
train_random(episodes)
end = time.time()
print(f"random: {(end-start)/episodes}")
# start = time.time()
# train_pg(env, episodes)
# end = time.time()
# print(f"PG: {(end-start)/episodes}")
# start = time.time()
# train_ac(env, episodes)
# end = time.time()
# print(f"AC: {(end-start)/episodes}")
# start = time.time()
# train_dqsn(env, episodes, sarsa = False)
# end = time.time()
# print(f"DQN: {(end-start)/episodes}")
# start = time.time()
# # train_dqsn_dcn(env, episodes, sarsa = False)
# train_dqsn_lstm(env, episodes, sarsa = False)
# end = time.time()
# print(f"DQN_LSTM: {(end-start)/episodes}")
# start = time.time()


# %%
