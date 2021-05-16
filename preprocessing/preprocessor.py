# import libiaries
import pandas as pd
import numpy as np

# for feature engineering
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, OrdinalEncoder
from sklearn.preprocessing import Normalizer, FunctionTransformer, PowerTransformer
from sklearn.impute import SimpleImputer
from feature_engine import imputation as imp
from feature_engine import discretisation as dsc
from feature_engine import encoding as ecd
from feature_engine.encoding import MeanEncoder


class DataPreprocessor(object):
    """Preprocess data with sklearn """
    def __init__(self):
        # Define a rename dictionary to map categories in Japanese to English
        self.rename_dict = {"week_x0_月":"week_mon", "week_x0_木":"week_tue",
                "week_x0_水":"week_wed", "week_x0_火":"week_thu",
                "week_x0_金":"week_fri",
                "remarks_x0_N/A":"no_remarks",
                "remarks_x0_お楽しみメニュー":"remarks_fun",
                "remarks_x0_スペシャルメニュー（800円）":"remarks_special",
                "remarks_x0_手作りの味":"remarks_homemade",
                "remarks_x0_料理長のこだわりメニュー":"remarks_chef",
                "remarks_x0_近隣に飲食店複合ビルオープン":"remarks_nearby",
                "remarks_x0_酢豚（28食）、カレー（85食）":"remarks_pork_curry",
                "remarks_x0_鶏のレモンペッパー焼（50食）、カレー（42食）": "remarks_chicken_curry",
                "event_x0_N/A":"no_event",
                "event_x0_キャリアアップ支援セミナー":"event_seminar",
                "event_x0_ママの会":"event_mama",
                "weather_x0_快晴":"weather_c_sunny",
                "weather_x0_晴れ":"weather_sunny",
                "weather_x0_曇":"weather_cloudy",
                "weather_x0_薄曇":"weather_s_cloudy",
                "weather_x0_雨":"weather_rainy",
                "weather_x0_雪":"weather_snow",
                "weather_x0_雷電":"weather_thunder",
                "name":"name_ord",
                "kcal":"kcal_dsc",
                "temperature":"temperature_dsc"}



    def set_preprocessors(self, unknown_value_for_name):
        # Initialize a preprocessor
        self.preprocessors = DataFrameMapper([
            # Missing data imputation for categorical variables
            (['event'], [SimpleImputer(strategy='constant', fill_value='N/A'),
                 OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')]),

            (['remarks'], [SimpleImputer(strategy='constant', fill_value='N/A'),
                   OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')]),
                   
            # encode categorical variables
            (['week'], OneHotEncoder(categories='auto',
                           sparse=False,
                           handle_unknown='ignore')),

            (['weather'], OneHotEncoder(categories='auto',
                           sparse=False,
                           handle_unknown='ignore')),
    
            # numerical data imputation
            (['kcal'], [SimpleImputer(strategy='median'),
                KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='uniform')
                ]),

            (['temperature'], [FunctionTransformer(lambda x: x**(1/1.2), validate=True),  #PowerTransformer(method='box-cox', standardize=False)
                      KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='uniform')
                      ]),

            # ordinal encoding
            (['name'], OrdinalEncoder(categories='auto',
                              handle_unknown='use_encoded_value',
                              unknown_value=unknown_value_for_name)
            ),
            
            ], input_df=True, df_out=True, default=None)




    def fit_transform(self, X_data, y_data):
        # Fit and apply preprocessors to training data
        X_data_p, y_data_p = X_data.copy(), y_data.copy()
        X_data_p = self.preprocessors.fit_transform(X_data_p, y_data_p)
        X_data_p.rename(columns=self.rename_dict, inplace=True)
        X_data_p = X_data_p.join(X_data[['name','temperature','kcal']])
        
        return X_data_p, y_data_p

    def transform(self, X_data):
        # Fit and apply preprocessors to training data
        X_data_p = X_data.copy()
        X_data_p = self.preprocessors.transform(X_data_p)
        X_data_p.rename(columns=self.rename_dict, inplace=True)
        X_data_p = X_data_p.join(X_data[['name','temperature','kcal']])
        
        return X_data_p


     