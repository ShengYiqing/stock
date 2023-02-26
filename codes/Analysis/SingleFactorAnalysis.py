
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

def single_factor_analysis(factor_name, start_date, end_date, white_threshold=0.618, value_type='factor_value'):
    factor_table_name = 'tfactor' + factor_name
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    
    sql = tools.generate_sql_y_x([factor_name], start_date, end_date, white_threshold)
    df = pd.read_sql(sql, engine)
    
    x = df.set_index(['trade_date', 'stock_code']).loc[:, factor_name].unstack()
    y = df.set_index(['trade_date', 'stock_code']).loc[:, 'r_daily'].unstack()
    
    tools.factor_analyse(x, y, 10, factor_name)
    
    # context_list = ['bp', 'ep', 'mc', 'momentum', 'sigma', 'tr', 'str', 'corrmarket']
    # context_list = ['mc']
    # pieces = 10
    # for context in context_list:
    #     sql = """
    #     select trade_date, stock_code, factor_value from factor.tfactor{context} 
    #     where trade_date >= {start_date} 
    #     and trade_date <= {end_date} """.format(context=context, start_date=start_date, end_date=end_date)
    #     context_df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).loc[:, 'factor_value'].unstack()
    #     context_df = DataFrame(context_df, index=x.index, columns=x.columns)
        
    #     context_quantile = context_df.rank(axis=1).div(context_df.notna().sum(1), axis=0)
    #     for i in range(pieces):
    #         x_tmp = x.copy()
    #         y_tmp = y.copy()
    #         x_tmp[~((i/pieces <= context_quantile)&(context_quantile <= (i+1)/pieces))] = np.nan
    #         y_tmp[~((i/pieces <= context_quantile)&(context_quantile <= (i+1)/pieces))] = np.nan
            
    #         tools.factor_analyse(x_tmp, y_tmp, 10, factor_name + '-context ' + context + ' ' + str(i))

if __name__ == '__main__':
    factor_name = 'dailytech'
    factors = [
        'momentum', 'volatility', 'skew', 'holo', 'corrmarket', 
        'tr', 'str', 
        'pvcorr', 
        ]
    factors = [
        'value', 'expectedquality', 
        
        'momentum', 'corrmarket', 
        'tr', 'str', 
        'pvcorr', 
        
        'oc', 'ca',
        ]
    start_date = '20120101'
    end_date = '20230301'
    white_threshold = 0.8
    value_type = 'preprocessed_factor_value'
    print(factor_name, start_date, end_date, white_threshold, value_type)
    single_factor_analysis(factor_name, start_date, end_date, white_threshold, value_type)
    white_threshold = 0
    value_type = 'preprocessed_factor_value'
    print(factor_name, start_date, end_date, white_threshold, value_type)
    single_factor_analysis(factor_name, start_date, end_date, white_threshold, value_type)