#%%
import time
import tensorflow as tf

from Models.Random import train_random
from Models.DQN import train_dqsn
from Models.PG import train_pg
from Models.TDAC import train_ac
from Models.DQN_lstm import train_dqsn_lstm

tf.get_logger().setLevel('ERROR')

# Configure GPU memory growth and eager execution for agents
try:
    tf.compat.v1.disable_eager_execution()
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    print("GPU Configured")
except Exception as e:
    print("No GPU's detected\n",e)


#%%

##TODO
# uniformise datetime slicing in data loader, OR timedelte OR amount of datapoints (considering market close at night), perhaps timedelta is better

episodes = 500

# start = time.time()
# train_random(episodes)
# end = time.time()
# print(f"\nrandom: {(end-start)/episodes}")
# start = time.time()
# train_pg(episodes)
# end = time.time()
# print(f"\nPG: {(end-start)/episodes}")
# start = time.time()
# train_ac(episodes)
# end = time.time()
# print(f"\nAC: {(end-start)/episodes}")
# start = time.time()
# train_dqsn(episodes, sarsa = False)
# end = time.time()
# print(f"\nDQN: {(end-start)/episodes}")
# start = time.time()
# train_dqsn_lstm(episodes, sarsa = False)
# end = time.time()
# print(f"\nDQN_LSTM: {(end-start)/episodes}")



# %%
import math

run_start = time.time()
timings = []
for e in range(episodes):
    ep_start_time = time.time()
    
    time.sleep(0.82547)
    
    timings.append(time.time()-ep_start_time)
    avg_time = sum(timings)/len(timings)
    m, s = divmod(math.floor(avg_time*(episodes-e)), 60)
    h, m = divmod(m, 60)
    
    print(f'\rEpisode: {e}/{episodes}, Time estimate: {math.floor(time.time() - run_start)}s/{math.floor(avg_time*500)}s => {h:d}:{m:02d}:{s:02d}.', end='', flush=True)
# %%
