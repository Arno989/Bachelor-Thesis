#%%
import matplotlib.pyplot as plt
import os, csv
import numpy as np
import pandas as pd
plt.style.use("seaborn-bright") # fivethirtyeight, classic, bmh

reward_data = {}
profit_data = {}

for i, hist_file in enumerate(os.listdir("./Records")):
    reward_data[hist_file], profit_data[hist_file] = [], []
    with open(os.path.join("./Records", hist_file), mode='r', newline='') as file:
        reader = csv.reader(file)
        for i, r in enumerate(reader):
            if i <= 208:
                reward_data[hist_file].append(float(r[0]))
                profit_data[hist_file].append(float(r[1]))
            else:
                break
            
reward_data = pd.DataFrame(reward_data)
profit_data = pd.DataFrame(profit_data)
            
# %%
plt.figure(figsize=(20,10))

for file in profit_data:
    plt.plot(file, data=profit_data)
plt.legend(loc='upper left')
plt.show

#%%
fig = plt.figure(figsize=(20,20))
gs = fig.add_gridspec(2, hspace=0)
(ax1, ax2) = gs.subplots(sharex=True)
for file in profit_data:
    ax1.plot(file, data=profit_data, )
ax1.legend(loc='upper left')

for file in reward_data:
    ax2.plot(file, data=reward_data)

for ax in fig.get_axes():
    ax.label_outer()

fig.show()
# %%
