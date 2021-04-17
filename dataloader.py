import os, datetime
import pandas as pd
import numpy as np
import gym, gym_anytrading

## TODO
# make per episode env reset
# change data to relative score against max profit
# randomise start date in data
# uniformise datetime slicing in data loader, OR timedelte OR amount of datapoints (considering market close at night), perhaps timedelta is better

class env_initialiser():
    def __init__(self):
        self.columnNames = ["Symbol", "Timestamp", "Open", "High", "Low", "Close", "Volume"]
        self.window_size = 60
        self.days = 2 # slice of data in in days
        self.data_len = self.days * 1440 # (1440 minutes in a day)
        self.data_list = os.listdir("./Data/Prices/")

    def init(self):
        df = pd.read_csv(f"./Data/Prices/{np.random.choice(self.data_list)}")
        
        # Slice and prepare data for environment
        df.columns = self.columnNames
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        ### Slicing on data_len
        ## define random start timestamp between start of data and end of data minus data_len, then take slice of data_len
        indexer = np.random.randint(0, len(df) - self.data_len)
        df = df[indexer: indexer + self.data_len]
        # print(df.iloc[0]["Symbol"], len(df))
        
        
        """ ### Slicing on date (reccommend 7 days)
        ## define random start timestamp between start of data and end of data minus data_len
        indexer = np.random.randint(0, len(df[df["Timestamp"] <= df["Timestamp"].max()-datetime.timedelta(days=self.days)]))
        first_timestamp = df.iloc[indexer]["Timestamp"]
        ## define timestamp of first timestamp + data_len
        last_timestamp = df[df["Timestamp"] >= (df.iloc[indexer]["Timestamp"] + datetime.timedelta(days=self.days))].iloc[0]["Timestamp"]
        
        # slice between start and end date
        df = df[(df["Timestamp"] >= first_timestamp) & (df["Timestamp"] <= last_timestamp)]
        # print(df.iloc[0]["Symbol"], len(df))
        """

        # step size = 60 minutes (make prediction every hour)
        start_index = self.window_size
        end_index = len(df)

        return gym.make('stocks-v0', df = df, window_size = self.window_size, frame_bound = (start_index, end_index))
        # print(f"Actions: {env.action_space.n}, Observation space: {env.observation_space.shape[0]}")