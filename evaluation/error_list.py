from .average_meter import AverageMeter
from math import sqrt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score

def RMSE(pred, target): return sqrt(MSE(target, pred))

class ErrorList(object):
    """"store errors of different models"""

    def __init__(self, model_keys, error_metrics):
        self.model_keys = model_keys
        self.error_metrics = error_metrics
        self.errors = dict()
        for key in self.model_keys:
            self.errors[key] = dict()
            self.errors[key]['train'] = {}
            self.errors[key]['valid'] = {}
            for m in self.error_metrics: 
                self.errors[key]['train'][m], self.errors[key]['valid'][m] = AverageMeter(), AverageMeter()
                self.errors[key]['train']['RMSE'], self.errors[key]['valid']['RMSE'] = AverageMeter(), AverageMeter()


    def update(self, model_key, y_train_t, y_train_preds, y_valid_t, y_valid_preds):
        for m in self.error_metrics:
            self.errors[model_key]['train'][m].update(eval(m)(y_train_t, y_train_preds.reshape(-1,1)), n=len(y_train_t))
            if m == 'r2_score':
                self.errors[model_key]['valid'][m].update(eval(m)(y_valid_t, y_valid_preds.reshape(-1,1)), n=len(y_valid_t))
            else:
                for i in range(len(y_valid_t)):
                    self.errors[model_key]['valid'][m].update(eval(m)(y_valid_t[i], y_valid_preds[i].reshape(-1,1)))
        self.errors[model_key]['valid']['RMSE'] = (self.errors[model_key]['valid']['MSE'].avg) ** 0.5
        self.errors[model_key]['train']['RMSE'] = (self.errors[model_key]['train']['MSE'].avg) ** 0.5

