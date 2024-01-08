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

def single_factor_analysis(df, factor_name, start_date, end_date):
    x = df.loc[:, factor_name].unstack()
    # x = tools.neutralize(x)
    y = df.loc[:, 'r'].unstack()
    
    tools.factor_analyse(x, y, 21, factor_name)
    
    
if __name__ == '__main__':
    factors = [
        'beta', 
        'mc', 
        'bp', 
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
    # factors = ['bp']
    start_date = '20120101'
    end_date = '20231231'
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    sql = tools.generate_sql_y_x(factors, start_date, end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    for factor_name in factors:
        print(factor_name, start_date, end_date)
        single_factor_analysis(df, factor_name, start_date, end_date)
        