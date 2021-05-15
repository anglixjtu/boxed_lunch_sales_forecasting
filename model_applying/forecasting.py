# import libiaries
import pandas as pd

# for the model
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor#, HistGradientBoostingRegressor


class Forecaster(object):
    """Perform time series forecast.
    	Arguments:
    """
    def __init__(self, model, t_lag=1, t_out=1):
        self.t_lag = t_lag
        self.t_out = t_out
        self.model = model

    
    def recursive_forecast(self, X_data):
        # make a recursive multi-step forecast
        # initialize outputs
        y_preds = pd.DataFrame(columns=['y(t)'] + ['y(t+%d)'%i for i in range(1, self.t_out)])

        for i in range(0, self.t_out):
            pred = self.model.predict(X_data.iloc[[i]])
            # store the predictions
            if i == 0:
                y_preds['y(t)'] = pred
            else:
                y_preds['y(t+%d)'%i] = pred
            # update historical targets/observations with the prediction
            if i < self.t_out-1:
                for j in range(1, self.t_out):
                    X_data['y(t-%d)'%(j+1)].iloc[i+1] = X_data['y(t-%d)'%(j)].iloc[i]
                    X_data['y(t-1)'].iloc[i+1] = pred

        return y_preds

    def train_infer_recursive(self, X_train, y_train, X_valid):
        # Train on the training set and recursively infer results on the validation/test sets

        self.model.fit(X_train, y_train)

        y_train_preds = self.model.predict(X_train)
        
        # recursive forecast
        y_valid_preds = self.recursive_forecast(X_valid)
    
        return y_train_preds, y_valid_preds.values.reshape(-1,)

    
