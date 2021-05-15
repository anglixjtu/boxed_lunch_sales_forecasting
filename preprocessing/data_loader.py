# import libiaries
import pandas as pd
import numpy as np


class DataLoader(object):
    """Load data and return basic information from the data set"""
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path)
    
    def __len__(self):
        return len(self.dataset)

    def get_types(self):
        # find categorical variables
        self.categorical = [var for var in self.dataset.columns if self.dataset[var].dtype=='O'] + ['payday']
        # find numerical variables
        self.numerical = [var for var in self.dataset.columns if var not in self.categorical and var != 'y']
        self.numerical = list(set(self.numerical)-set(['datetime', 'soldout', 'payday']))
        # find discrete variables (categories less than 31)
        self.discrete = []
        for var in self.numerical:
            if len(self.dataset[var].unique()) <= 31:
                self.discrete.append(var)

        # find datetime variables
        self.datetime = ['datetime']
        
        return self.categorical, self.numerical, self.discrete, self.datetime

    def preprocess(self):
        # convert string to datetime
        self.dataset['datetime'] = pd.to_datetime(self.dataset['datetime'])
        self.dataset['precipitation'] = self.dataset['precipitation'].replace('--', 0)
        self.dataset['precipitation'] = self.dataset['precipitation'].astype(np.float64)
        self.dataset.loc[self.dataset['payday']!=1, 'payday'] = np.int64(0)

        return self.dataset



  
        
