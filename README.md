# Boxed Lunch Sales Forecasting

This repository provides a solution for a practice problem on SIGNATE:  [Boxed Lunch Sales Forecasting](https://signate.jp/competitions/24
). The sales of boxed lunch at a certain company O in Chiyoda-ku are possibly related to many factors such as the menu, the weather, or the date. Our goal is to forecast the sales of the boxed lunch.

##
 Dependency

- Python 3.9.2
- Numpy 1.19.2
- Pandas 1.2.4
- Scikit-learn 0.24.1
- Feature-engine 1.0.2
- Sklearn-pandas 2.1.0
- Matplotlib 3.4.1
- Seaborn 0.11.1

## 1. Exploratory Data Analysis

To perform exploratory data analysis and check the results, run code in [exploratory_data_analysis](notebooks/exploratory_data_analysis.ipynb).

## 2. Machine Learning Modeling

To perform machine learning modeling, you can execute  

1. Jupyter Notebook code in [modeling](notebooks/modeling.ipynb), or

2. Python code in [train_valid_models.py](train_valid_models.py).

## Reference

[1] [Walk-forward Validation](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting)

[2] [Recursive Method for Multi-step Forecasting](https://machinelearningmastery.com/multi-step-time-series-forecasting/)



