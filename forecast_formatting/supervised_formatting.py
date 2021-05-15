from model_construction_validation import train_builtin_forecast_models
import pandas as pd

class SupervisedFormatter(object):
    """Frame a time series as a supervised learning dataset.
    	Arguments:
		t_in: Number(Length) of lag observations as input (X).
		t_out: Number(Length) of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
    """

    def make_lag_variables(self, data, t_lag, t_out):
        cols, names = list(), list()
        # input sequence (t-n, ... t)
        for i in range(t_lag, -1, -1):
            cols.append(data.shift(i))
            if i == 0:
                names += [('%s(t)' % (j)) for j in data.columns]
            else:
                names += [('%s(t-%d)' % (j, i)) for j in data.columns]
	    # forecast sequence (t+1, ... t+n)
        for i in range(1, t_out):
            cols.append(data.shift(-i))
            names += [('%s(t+%d)' % (j, i)) for j in data.columns]
	    # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        return agg

    def format_train(self, X, y, t_lag, t_out=1):
        # function to create log variables and drop rows with nan values
        X_t = self.make_lag_variables(X, t_lag=t_lag, t_out=0).iloc[t_lag:]
        y_t = self.make_lag_variables(y, t_lag=t_lag, t_out=t_out).iloc[t_lag:]
        # separate features and targets from y
        y_f = y_t[['y(t-%d)' % i for i in range(1, t_lag+1)]]
        y_t = y_t[['y(t)']+['y(t+%d)' % i for i in range(1, t_out)]]
        return X_t, y_f, y_t

    def format_valid(self, X, y, t_lag, t_out=1):
        # function to create log variables and drop rows with nan values
        X_t = self.make_lag_variables(X, t_lag=t_lag, t_out=0).iloc[t_lag:]
        y_t = self.make_lag_variables(y, t_lag=t_lag, t_out=t_out)
        if t_out==1:
            y_t = y_t.iloc[[-1]]
        else:
            y_t = y_t.iloc[t_lag:-(t_out-1)]
        # separate features and targets from y
        y_f = y_t[['y(t-%d)' % i for i in range(1, t_lag+1)]]
        y_t = y_t[['y(t)']+['y(t+%d)' % i for i in range(1, t_out)]]
        return X_t, y_f, y_t



    
