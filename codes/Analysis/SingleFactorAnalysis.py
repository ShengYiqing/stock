import datetime
import os
import sys
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
from sqlalchemy import create_engine

def single_factor_analysis(factor_name, start_date, end_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    
    sql = tools.generate_sql_y_x([factor_name], start_date, end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    # df.loc[:, ['reversal', 'pvcorr']] = (-1) * df.loc[:, ['reversal', 'pvcorr']]
    # x = df.groupby('trade_date').rank(pct=True).loc[:, ['reversal', 'wmdaily', 'pvcorr', 'beta']].mean(1).unstack()
    x = df.loc[:, factor_name].unstack()
    # x = tools.neutralize(x)
    y = df.loc[:, 'r_d'].unstack()
    
    tools.factor_analyse(x, y, 21, factor_name)
    
    
if __name__ == '__main__':
    factor_name = 'hfbeta'
    
    factors = [
        'hfbeta', 'hfcorrmarket', 
        'hfsigma', 'hfskew', 'hfhl', 
        'hfetr', 'hfttr', 'hfutr', 
        'hfspread', 'hfimbalance', 
        'hfpvcorr', 'hfrvcorr', 'hfpscorr', 'hfpicorr', 
        'hfintradaymomentum', 
        'hfcallauctionmomentum', 'hfbluff'
        ]
    factors = [
        'sigmad',
        'beta', 
        'crhl', 'crshl', 
        'crv', 'crsv', 
        'crsm',
        ]
    factors = ['tr']
    factor_dic = {
        'crtrd': -1,
        'crhld': -1, 
        'crco': 1, 
        'croc': 1,
        'crhl2o': 2,
        
        'crsmvold': 2, 
        }
    factors = list(factor_dic.keys())# + ['cxx']
    # factors = ['quality', 'expectation']
    factors = ['beta', 'reversal', 'cxx', 'hftech']
    # factors = ['cxx']
    for factor_name in factors:
        start_date = '20210101'
        end_date = '20230730'
        print(factor_name, start_date, end_date)
        single_factor_analysis(factor_name, start_date, end_date)
        