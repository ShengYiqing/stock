# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:47:43 2021

@author: admin
"""

import os
import sys
import datetime
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)

import Global_Config as gc
import tools
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sqlalchemy import create_engine

import multiprocessing as mp

def f(factor_name, white_threshold, start_date, end_date):
    factor_table_name = 'tfactor' + factor_name

    print(factor_table_name)
    print(white_threshold)
    print(datetime.datetime.now())
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    
    sql = tools.generate_sql_y_x([factor_name], start_date, end_date, white_threshold)
    
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    y_d = df.loc[:, 'r_daily'].unstack()
    y_w = df.loc[:, 'r_weekly'].unstack()
    y_m = df.loc[:, 'r_monthly'].unstack()
    x = df.loc[:, factor_name].unstack()
    
    ic_d = x.corrwith(y_d, axis=1)
    ic_w = x.corrwith(y_w, axis=1)
    ic_m = x.corrwith(y_m, axis=1)
    
    rank_ic_d = x.corrwith(y_d, axis=1, method='spearman')
    rank_ic_w = x.corrwith(y_w, axis=1, method='spearman')
    rank_ic_m = x.corrwith(y_m, axis=1, method='spearman')
    
    df = DataFrame({'IC_D':ic_d, 'RANK_IC_D':rank_ic_d, 
                    'IC_W':ic_w, 'RANK_IC_W':rank_ic_w, 
                    'IC_M':ic_m, 'RANK_IC_M':rank_ic_m, 
                    })
    df.loc[:, 'white_threshold'] = '%s'%white_threshold
    df.loc[:, 'is_neutral'] = 0
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.loc[:, 'FACTOR_NAME'] = factor_table_name[7:]

    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/ic?charset=utf8")
    df.to_sql('tdailyic', engine, schema='ic', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

    # sql = tools.generate_sql_y_x([factor_name], start_date, end_date, white_threshold, factor_value_type='neutral_factor_value')
    
    # df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    # y_d = df.loc[:, 'r_daily'].unstack()
    # y_w = df.loc[:, 'r_weekly'].unstack()
    # y_m = df.loc[:, 'r_monthly'].unstack()
    # x = df.loc[:, factor_name].unstack()
    
    # ic_d = x.corrwith(y_d, axis=1)
    # ic_w = x.corrwith(y_w, axis=1)
    # ic_m = x.corrwith(y_m, axis=1)
    
    # rank_ic_d = x.corrwith(y_d, axis=1, method='spearman')
    # rank_ic_w = x.corrwith(y_w, axis=1, method='spearman')
    # rank_ic_m = x.corrwith(y_m, axis=1, method='spearman')
    
    # df = DataFrame({'IC_D':ic_d, 'RANK_IC_D':rank_ic_d, 
    #                 'IC_W':ic_w, 'RANK_IC_W':rank_ic_w, 
    #                 'IC_M':ic_m, 'RANK_IC_M':rank_ic_m, 
    #                 })
    # df.loc[:, 'white_threshold'] = '%s'%white_threshold
    # df.loc[:, 'is_neutral'] = 1
    # df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    # df.loc[:, 'FACTOR_NAME'] = factor_table_name[7:]

    # engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/ic?charset=utf8")
    # df.to_sql('tdailyic', engine, schema='ic', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

    
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    # start_date = '20100101'
    
    factors = [
        'mc', 'bp', 
        'quality', 'value', 
        'momentum', 'volatility', 'liquidity', 'corrmarket',
        'dailytech', 'hftech', 
        ]
    white_thresholds = [0, 0.2, 1-0.618, 0.618, 0.8]
    # white_thresholds = [0.2]
    pool = mp.Pool(4)
    for factor in factors:
        for white_threshold in white_thresholds:
            pool.apply_async(func=f, args=(factor, white_threshold, start_date, end_date))
    pool.close()
    pool.join()