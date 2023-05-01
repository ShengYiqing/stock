
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

def single_factor_analysis(factor_name, start_date, end_date, neutral_list):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    
    sql = tools.generate_sql_y_x([factor_name], start_date, end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    
    x = df.loc[:, factor_name].unstack()
    y = df.loc[:, 'r_daily'].unstack()
    
    tools.factor_analyse(x, y, 10, factor_name)
    if neutral_list == None:
        x = df.loc[:, factor_name].unstack()
    else:
        if 'ind' in neutral_list:
            ind = 'l3'
        neutral_list = [i for i in neutral_list if i != 'ind']
        x = tools.neutralize(df.loc[:, factor_name].unstack(), neutral_list, ind)
    
    x_n = tools.neutralize(x, neutral_list, ind)
    
    tools.factor_analyse(x_n, y, 10, factor_name)
    
    
if __name__ == '__main__':
    factor_name = 'sigma'
    neutral_list = ['ind', 'mc']
    neutral_list = ['ind', 'mc', 'sigma']
    neutral_list = None
    factors = [
        'quality', 'value', 
        'momentum', 'str',
        'dailytech', 'hftech', 
        ]
    
    start_date = '20120101'
    end_date = '20230414'
    print(factor_name, start_date, end_date, neutral_list)
    single_factor_analysis(factor_name, start_date, end_date, neutral_list)
    