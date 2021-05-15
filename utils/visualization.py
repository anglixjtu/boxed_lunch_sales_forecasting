import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats


class HistQQVisualizer(object):
    """plot the histograms to have a quick look at the variable distribution
     histogram and Q-Q plots"""

    def __init__(self, figsize=(12, 3)):
        self.figsize = figsize

    def show(self, data, variable):
        # function to plot a histogram and a Q-Q plot
        # side by side, for a certain variable

        plt.figure(figsize=self.figsize)                              
        plt.subplot(1, 2, 1)
        data[variable].hist(bins=30)

        plt.subplot(1, 2, 2)
        stats.probplot(data[variable], dist="norm", plot=plt)

        plt.show()


class CurveVisualizer(object):
    """plot the curves of predictions with groundtruth and curves of errors"""

    def __init__(self, colors, x_label='datetime', y_label='y', figsize=(12, 3), labelsize=15, fontsize=10, tick_labelsize=20, linewidth=2, markersize=600 ):
        self.colors = colors
        self.x_label, self.y_label = x_label, y_label
        self.figsize, self.labelsize, self.fontsize, self.tick_labelsize =\
         figsize, labelsize, fontsize, tick_labelsize
        self.linewidth = linewidth
        self.markersize = markersize

    def set_text(self, ax, ylabel):
        ax.set_ylabel(ylabel, fontsize=self.labelsize)
        ax.legend(fontsize=self.fontsize)
        plt.setp(ax.get_xticklabels(), rotation=-25, ha="left",
         rotation_mode="anchor")
        ax.tick_params(labelsize=self.labelsize)
        ax.grid(axis='x')


    def show_predictions(self, data, pred, pred_train=None, show_marks=('remarks','お楽しみメニュー' ), show=True, savepath=None):
        # function to plot curves of predictions and groundtruth
        
        # extract groundtruth data
        x = data[self.x_label]
        y = data[self.y_label]
   
        test_size = len(pred[list(pred.keys())[0]])
    
        if pred_train is None: # show without training predictions
           x, y = x[-test_size:], y[-test_size:]
           fig, ax = plt.subplots(figsize=(self.figsize[0]//2, self.figsize[1]) )
        else:
            fig, ax = plt.subplots(figsize=self.figsize)

        # plot groundtruth
        ax.plot(x, y ,linewidth=self.linewidth, label='groundtruth', color=self.colors[0], linestyle='--')

        # plot predictions
        i = 1
        for model_keys in pred.keys():
            if pred_train: 
                yhat = list(pred_train[model_keys]) + list(pred[model_keys]) # concatenate training predictions
            else:
                yhat = pred[model_keys]
            ax.plot(x, yhat ,linewidth=self.linewidth, label=model_keys, color=self.colors[i])
            i += 1

        self.set_text(ax, ylabel='sales')

        # show marks
        if show_marks:
            remarks_color = self.colors[-1]
            data_with_remarks = data[[self.x_label, self.y_label]][data[show_marks[0]]==show_marks[1]]
            ax.scatter(data_with_remarks[self.x_label], data_with_remarks[self.y_label], self.markersize, color=remarks_color,marker='*')

        if show:
            plt.show()
    
        if savepath:
            fig.savefig(savepath)


    def show_metric(self, data, metric, show=True, metric_key='MAE', savepath=None):
        # function to plot curves of error metrics (only on the test/validation set
        x = data[self.x_label]
        
        fig, ax = plt.subplots(figsize=(self.figsize[0]//2, self.figsize[1]))

        test_size = metric[list(metric.keys())[0]]['valid']['MAE'].count
        
        i = 1
        for model_key in metric.keys():
            ax.plot(x[-test_size:], metric[model_key]['valid'][metric_key].val_list,linewidth=self.linewidth, label=model_key, color=self.colors[i])
            i += 1
        self.set_text(ax, ylabel=metric_key)  
        
        if show:
            plt.show()
            
        if savepath:
            fig.savefig(savepath)


class BarVisualizer(object):
    """plot the bars of errors"""

    def __init__(self, colors, x_label='datetime', y_label='y', figsize=(9, 6), labelsize=15, fontsize=10, tick_labelsize=20, width=0.35, padding=3 ):
        self.colors = colors
        self.x_label, self.y_label = x_label, y_label
        self.figsize, self.labelsize, self.fontsize, self.tick_labelsize =\
         figsize, labelsize, fontsize, tick_labelsize
        self.width = width
        self.padding = padding


    def set_text(self, ax, x, metric_key, labels):
        # Set text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(metric_key, fontsize=self.labelsize)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=self.fontsize)
        ax.tick_params(labelsize=self.labelsize)


    def show_train_valid_errors(self, metric, metric_key='MAE', show=True, savepath=None):
        # function to plot of error metrics on both validation and training sets
        fig, ax = plt.subplots(figsize=self.figsize)
        labels = [model_key for model_key in metric.keys()]
        labels, train_errs, valid_errs = [], [], []
        for model_key in metric.keys():
            labels += [model_key]
            if metric_key == 'RMSE':
                train_errs += [round(metric[model_key]['train'][metric_key], 2)]
                valid_errs += [round(metric[model_key]['valid'][metric_key], 2)]
            else:
                train_errs += [round(metric[model_key]['train'][metric_key].avg, 2)]
                valid_errs += [round(metric[model_key]['valid'][metric_key].avg, 2)]

        x = np.arange(len(labels))  # the label locations

        rects1 = ax.bar(x - self.width/2, train_errs, self.width, label='training', color=self.colors[0])
        rects2 = ax.bar(x + self.width/2, valid_errs, self.width, label='validation', color=self.colors[1])

                # label the hight of bars
        ax.bar_label(rects1, padding=self.padding)
        ax.bar_label(rects2, padding=self.padding)

        self.set_text(ax, x, metric_key, labels)
        
        fig.tight_layout()
        
        if show:
            plt.show()
    
        if savepath:
            fig.savefig(savepath)



    def show_predict_forecast_errors(self, metric_p, metric_f, metric_key='MAE', show=True, savepath=None):
        # function to plot of error metrics of prediction and forecast models on both validation and training sets
        fig, ax = plt.subplots(figsize=self.figsize)
        labels = [model_key for model_key in metric_p.keys()]
        labels, predict_train_errs, forecast_train_errs, predict_valid_errs, forecast_valid_errs = [], [], [], [], []
        for model_key in metric_p.keys():
            labels += [model_key]
            if metric_key == 'RMSE':
                predict_train_errs += [round(metric_p[model_key]['train'][metric_key], 2)]
                predict_valid_errs += [round(metric_p[model_key]['valid'][metric_key], 2)]
                forecast_train_errs += [round(metric_f[model_key]['train'][metric_key], 2)]
                forecast_valid_errs += [round(metric_f[model_key]['valid'][metric_key], 2)]
            else:
                predict_train_errs += [round(metric_p[model_key]['train'][metric_key].avg, 2)]
                predict_valid_errs += [round(metric_p[model_key]['valid'][metric_key].avg, 2)]
                forecast_train_errs += [round(metric_f[model_key]['train'][metric_key].avg, 2)]
                forecast_valid_errs += [round(metric_f[model_key]['valid'][metric_key].avg, 2)]

        x = np.arange(len(labels))  # the label locations

        rects1 = ax.bar(x - 3*self.width/4, predict_train_errs, self.width/2, label='Prediction(train)', color=self.colors[0])
        rects2 = ax.bar(x - self.width/4, forecast_train_errs, self.width/2, label='Forecasting (tain)', color=self.colors[1])
        rects3 = ax.bar(x + self.width/4, predict_valid_errs, self.width/2, label='Prediction (validation)', color='w', hatch='//', edgecolor=self.colors[0])
        rects4 = ax.bar(x + 3*self.width/4, forecast_valid_errs, self.width/2, label='Forecasting (validation)', color='w', hatch='//', edgecolor=self.colors[1])

        # label the hight of bars
        for rects in [rects1, rects2, rects3, rects4]:
            ax.bar_label(rects, padding=self.padding)
  
        self.set_text(ax, x, metric_key, labels)
        
        fig.tight_layout()
        
        if show:
            plt.show()
    
        if savepath:
            fig.savefig(savepath)

        