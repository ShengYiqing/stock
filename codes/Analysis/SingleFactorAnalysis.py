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
    x = df.loc[:, factor_name].unstack()
    y = df.loc[:, 'r_d'].unstack()
    
    tools.factor_analyse(x, y, 10, factor_name)
    
    
if __name__ == '__main__':
    factors = [
        'beta', 
        'jump', 
        'reversal', 
        'momentum',  
        'seasonality',
        'skew', 
        'crhl', 
        'cphl',
        ]
    factors = [
        'operation', 
        'gross', 
        'core', 
        'profitability', 
        'cash', 
        'growth', 
        'stability', 
        'quality'
        ]
    factors = ['bp']
    for factor_name in factors:
        start_date = '20120101'
        end_date = '20231030'
        print(factor_name, start_date, end_date)
        single_factor_analysis(factor_name, start_date, end_date)
        