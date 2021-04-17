import os, datetime
import pandas as pd
import numpy as np
import gym, gym_anytrading


class env_initialiser():
    def __init__(self):
        self.columnNames = ["Symbol", "Timestamp", "Open", "High", "Low", "Close", "Volume"]
        self.window_size = 60
        self.data_len = 7 # slice of data in in days (1 min interval) (1 week is 2100 datapoints)
        self.data_list = os.listdir("./Data/Prices/")

    def init(self):
        df = pd.read_csv(f"./Data/Prices/{np.random.choice(self.data_list)}")
        
        # Slice and prepare data for environment
        df.columns = self.columnNames
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        ## define random start date between start of data and end of data minus data_len
        indexer = np.random.randint(0, len(df[df["Timestamp"] <= df["Timestamp"].max()-datetime.timedelta(days=self.data_len)]))
        ## slice all smaller than (indexer + data len), then slice from indexer to end
        df = df[df["Timestamp"] <= (df.iloc[indexer]["Timestamp"] + datetime.timedelta(days=self.data_len))][indexer:]

        # step size = 60 minutes (make prediction every hour)
        start_index = self.window_size
        end_index = len(df)

        return gym.make('stocks-v0', df = df, window_size = self.window_size, frame_bound = (start_index, end_index))
        # print(f"Actions: {env.action_space.n}, Observation space: {env.observation_space.shape[0]}")