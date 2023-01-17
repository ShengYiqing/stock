
import datetime
import os
import sys
import pickle
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import tools
import Global_Config as gc
import pickle

class Model:
    def __init__(self, model_name, label_weight, factor_name_list, start_date, end_date, rolling_n, rolling_weight):
        self.model_name = model_name
        self.label_weight = label_weight
        self.factor_name_list = factor_name_list
        self.start_date = start_date
        self.end_date = end_date
        self.rolling_n = rolling_n
        self.rolling_weight = rolling_weight
        trade_cal = pd.read_csv('%s/TradeCalData/TradeCal.csv', index_col=[0], parse_dates=[0])
        trade_cal = trade_cal.loc[trade_cal.loc[:, 'is_open']==1, :]
        if start_date:
            trade_cal = trade_cal[trade_cal.index > start_date]
        if end_date:
            trade_cal = trade_cal[trade_cal.index < end_date]
        trade_cal = trade_cal.index
        self.trade_cal = trade_cal

    def get_X(self):
        factor_list = []
        for factor_name in self.factor_name_list:
            factor_list.append(pd.read_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, factor_name), index_col=[0], parse_dates=[0]))
        return factor_list
    
    def get_y(self):
        y1 = pd.read_csv('%s/Data/y1.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0])
        y2 = pd.read_csv('%s/Data/y2.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0])
        y3 = pd.read_csv('%s/Data/y3.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0])
        y4 = pd.read_csv('%s/Data/y4.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0])
        y5 = pd.read_csv('%s/Data/y5.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0])
        
        if self.start_date:
            y1 = y1.loc[y1.index >= self.start_date, :]
            y2 = y2.loc[y2.index >= self.start_date, :]
            y3 = y3.loc[y3.index >= self.start_date, :]
            y4 = y4.loc[y4.index >= self.start_date, :]
            y5 = y5.loc[y5.index >= self.start_date, :]
            
        if self.end_date:
            y1 = y1.loc[y1.index <= self.end_date, :]
            y2 = y2.loc[y2.index <= self.end_date, :]
            y3 = y3.loc[y3.index <= self.end_date, :]
            y4 = y4.loc[y4.index <= self.end_date, :]
            y5 = y5.loc[y5.index <= self.end_date, :]
        
        y_list = [y1, y2, y3, y4, y5]
        
        y = 0 * y1
        
        for i in range(len(self.label_weight)):
            y = y + self.label_weight[i] * y_list[i]
        
        return y
    
    def generate_data(self, date):
        factor_list = self.factor_list
        
        y = None
        X = None
        return y, X
    
    def generate_model(self):
        for date in self.trade_cal:
            y, X = self.generate_data(date)
            model = self.fit()
            with open('%s/Results/%s/%s.pkl'%(gc.MODEL_PATH, self.model_name, date)) as f:
                pickle.dump(model, f)
                


    def fit(self):
        model = None
        return model
    
    def predict(self):
        pass