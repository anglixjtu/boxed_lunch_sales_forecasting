# import libiaries
import pandas as pd

class Predictor(object):
    """Perform time series prediction.
    	Arguments:
    """
    def __init__(self, model):
        self.model = model


    def train_infer(self, X_train, y_train, X_valid):
        # Train on the training set and infer results on the validation/test sets
  
        self.model.fit(X_train, y_train)

        y_train_preds = self.model.predict(X_train)
        y_valid_preds = self.model.predict(X_valid)
    
        return y_train_preds, y_valid_preds