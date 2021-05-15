# import libraries
import pandas as pd
import numpy as np

# for the model
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import *
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor#, HistGradientBoostingRegressor

# Import user defined libraries
from preprocessing.data_loader import DataLoader
from preprocessing.preprocessor import DataPreprocessor
from forecast_formatting.supervised_formatting import SupervisedFormatter
from model_applying.forecasting import Forecaster
from model_applying.prediction import Predictor
from utils.visualization import CurveVisualizer, BarVisualizer
from evaluation.error_list import ErrorList

# set args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', dest='data_dir', type=str, default="./data/",help='Directory for loading input data')
parser.add_argument('--outdir', dest='output_dir', type=str, default="./outputs/",help='Directory for saving output files')
parser.add_argument('--past_step', dest='past_step', type=int, default=2, help='The length of lagged variables')
parser.add_argument('--out_step', dest='out_step', type=int, default=2, help='The length of outputs')


# prepare a list of ml models
def get_models(models=dict()):
	# models
    max_depth = 10
    n_estimators = 50
    models['LR'] = LinearRegression()
    models['RF'] = RandomForestRegressor(n_estimators=20,max_depth=10,random_state=0)
    models['ST'] = StackingRegressor(estimators=[('rf', RandomForestRegressor(n_estimators=20,max_depth=10, random_state=0)),
        #('dt', DecisionTreeRegressor(max_depth=4)),
        ('lr', Lasso()),
        #('lr1', LinearRegression()),
                                                 
                                                 
                                                 ], 
                                     final_estimator=RandomForestRegressor(n_estimators=20,max_depth=4, random_state=0),
                                     passthrough=True)#DecisionTreeRegressor(max_depth=max_depth)
    
    #models['GB'] = GradientBoostingRegressor(random_state=0, n_estimators=20, max_depth=10, learning_rate=0.6)
    #models['AB'] = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth), random_state=0, n_estimators=n_estimators)
    #models['BG'] = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimators, random_state=0)
    #models['DT'] = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    print('Defined %d models' % len(models))
    return models


def train_models(data, numerical, valid_step=1, past_step=2, validation_size=20, error_metrics={'MSE', 'MAE'}):
    # keep the length of training and test set
    data_size = len(data)

    # initialize models, predictions, and errors
    model_list = get_models()
    error_list = ErrorList(model_list.keys(), error_metrics)
    predict_list, predict_list_train = dict(), dict()

    # train and validate each models
    for model_key in model_list.keys():
        model = model_list[model_key]
        predict_list[model_key], predict_list_train[model_key] = [], []

        # perform walk-forward validation
        min_train_size = data_size - validation_size
        for split_point in range(min_train_size, data_size, valid_step):
            # split training and validation sets with sliding windows
            X_train = data.drop('y', axis=1).iloc[:split_point]
            y_train = data[['y']].iloc[:split_point]
            X_valid = data.drop('y', axis=1).iloc[split_point-past_step:split_point+valid_step] #slice for forecasting is longer than that for prediction
            y_valid = data[['y']].iloc[split_point-past_step:split_point+valid_step]

            # preprocess data
            feature_processor = DataPreprocessor()
            feature_processor.set_preprocessors(len(X_train['name'].unique()))
            X_train, y_train = feature_processor.fit_transform(X_train, y_train)
            X_valid = feature_processor.transform(X_valid)


            # create lag variables
            formatter = SupervisedFormatter()
            X_train_t, y_train_f, y_train_t = formatter.format_train(X_train, y_train, t_lag=past_step, t_out=1)
            X_valid_t, y_valid_f, y_valid_t = formatter.format_valid(X_valid, y_valid, t_lag=past_step, t_out=valid_step)

            # feature_selection    
            week_dummies = [var for var in X_train.columns if 'week' in var]                     
            remarks_dummies = [var for var in X_train.columns if 'remarks' in var]
            event_dummies = [var for var in X_train.columns if 'event' in var]
            weather_dummies = [var for var in X_train.columns if 'weather' in var]
            numerical_dsc = [var + '_dsc' for var in numerical]

            selected_features =  ['temperature_dsc', 'remarks_fun']

            selected_features_t = ['%s(t)'%var for var in selected_features]
            for i in range(0, past_step):
                selected_features_t += ['%s(t-%d)'%(var, i+1) for var in ['temperature_dsc']]#'temperature_dsc'

            X_train_t = X_train_t[selected_features_t]
            X_valid_t = X_valid_t[selected_features_t]

            X_train_t, X_valid_t = X_train_t.astype(float), X_valid_t.astype(float)

            X_train_t = X_train_t.join(y_train_f)
            X_valid_t = X_valid_t.join(y_valid_f)

            
            # train and evaluate models
            if past_step == 0:
                predictor = Predictor(model)
                y_train_preds, y_valid_preds = predictor.train_infer(X_train_t, y_train_t.values.reshape(-1,),  X_valid_t)
            else:
                predictor = Forecaster(model, t_lag=past_step, t_out=valid_step)
                y_train_preds, y_valid_preds = predictor.train_infer_recursive(X_train_t, y_train_t.values.reshape(-1,),  X_valid_t)


            y_valid_t = y_valid_t.values.reshape(-1,1)

            # evaluate performance of models
            error_list.update(model_key, y_train_t, y_train_preds, y_valid_t, y_valid_preds)
            
            # store predicts of models
            predict_list[model_key] += list(y_valid_preds)

        # store training predicts of models
        predict_list_train[model_key] = y_train_preds.reshape(-1,)[:min_train_size-past_step]

        # output evaluation results
        for m in error_metrics:
            print('Model ({}) -- train {}: {}'.format(model_key, m, error_list.errors[model_key]['train'][m].avg))
        print('Model ({}) -- train {}: {}'.format(model_key, 'RMSE', error_list.errors[model_key]['train']['RMSE']))
        print()
        for m in error_metrics:
            print('Model ({}) -- valid {}: {}'.format(model_key, m, error_list.errors[model_key]['valid'][m].avg))
        print('Model ({}) -- valid {}: {}'.format(model_key, 'RMSE', error_list.errors[model_key]['valid']['RMSE']))
        print("-"*70)

    return predict_list_train, predict_list, error_list.errors


def main(args):

    # set directory
    output_dir, data_dir = args.output_dir, args.data_dir

    # Set parameters
    past_step, valid_step = args.past_step, args.out_step # the length of lagging, 0 for predicting, >0 for forecasting; the length of outputs for forecasting
    validation_size = 20 # keep the last (validation_size) of items for validation
    error_metrics = {'MSE', 'MAE'}

    # Load datasets
    training_path = data_dir + "train.csv"
    test_path = data_dir + "test.csv"
    sample_path = data_dir + "sample.csv"
    loader = DataLoader(training_path)
    data = loader.preprocess()


    # find categorical, numerical, and discrete variables
    categorical, numerical, discrete, datetime = loader.get_types()

    # start train and evaluate models
    # test prediction models
    prediction_train, prediction, p_errors = train_models(data, numerical, valid_step=valid_step, past_step=0)

    # test forecast models
    forecast_train, forecast, f_errors = train_models(data, numerical, valid_step=valid_step, past_step=past_step)

    # visualize results
    colors = [[68/255, 94/255, 126/255],
              [226/255, 117/255, 0/255],
              [0/255, 152/255, 166/255],
              [255/255, 124/255, 143/255],
              [153/255, 102/255, 0/255],
              [17/255, 137/255, 123/255],
              [134/255, 20/255, 112/255]]

    '''curve_visualizer = CurveVisualizer(colors)
    curve_visualizer.show_predictions(data, prediction, pred_train=prediction_train, show_marks=('remarks','お楽しみメニュー' ), show=True, savepath=None)

    curve_visualizer.show_metric(data, p_errors, show=True, metric_key='MAE', savepath=None)'''

    bar_visualizer = BarVisualizer(colors)
    '''bar_visualizer.show_train_valid_errors(p_errors, metric_key='MAE', show=True, savepath=None)

    bar_visualizer.show_train_valid_errors(p_errors, metric_key='RMSE', show=True, savepath=None)'''

    bar_visualizer.show_predict_forecast_errors(p_errors, f_errors, metric_key='RMSE', show=True, savepath=None)





    check_point = 295




if __name__ == "__main__":
    args = parser.parse_args()
    main(args)