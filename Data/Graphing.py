#%%
import matplotlib.pyplot as plt
import os, csv
import numpy as np
import pandas as pd

plt.rcParams.update(plt.rcParamsDefault)
# print(style.available)
plt.style.use("seaborn-bright") # fivethirtyeight, classic, bmh, seaborn-bright

reward_data = {}
profit_data = {}
profit_percentage = {}
profit_max = {}

for i, hist_file in enumerate(['Random.csv','PG.csv', 'TDAC.csv', 'DQN.csv']): # enumerate(os.listdir("./Records")) # , 'PG.csv', 'TDAC.csv', 'DQN.csv', 'DQN_lstm-b1.csv', 'max_profit.csv'
    reward_data[hist_file], profit_data[hist_file] = [], []
    with open(os.path.join("./Training Records", hist_file), mode='r', newline='') as file:
        reader = csv.reader(file)
        for i, r in enumerate(reader):
            if i < 500:
                # reward_data[hist_file].append(float(r[0]))
                profit_data[hist_file].append(float(r[2]))
                # profit_percentage[hist_file].append(float(r[2]))
                # profit_max[hist_file].append(float(r[3]))
            else:
                break
            
# reward_data = pd.DataFrame.from_dict(reward_data, orient='index')
# reward_data = reward_data.transpose()
profit_data = pd.DataFrame.from_dict(profit_data, orient='index')
profit_data = profit_data.transpose()
# profit_percentage = pd.DataFrame.from_dict(profit_percentage, orient='index')
# profit_percentage = profit_percentage.transpose()
# profit_max = pd.DataFrame.from_dict(profit_max, orient='index')
# profit_max = profit_max.transpose()

## remove all outliers
from scipy import stats
profit_data = profit_data[(np.abs(stats.zscore(profit_data)) < 3).all(axis=1)]

plt.figure(figsize=(20,10))

for file in profit_data:
    plt.plot(file, data=profit_data)
plt.legend(loc='upper left')
plt.ylabel('% of max reward')
plt.xlabel('episode')
plt.show()
#%%

