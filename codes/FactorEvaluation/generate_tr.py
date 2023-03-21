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
    start_date_sql = tools.trade_date_shift(start_date, 60)
    sql = tools.generate_sql_y_x([factor_name], start_date_sql, end_date, white_threshold, factor_value_type='factor_value')
    
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    df = df.loc[:, [factor_name]].dropna(subset=factor_name)
    
    x = tools.standardize(tools.winsorize(df.loc[:, factor_name].unstack()))
    
    tr_d = x.corrwith(x.shift(), axis=1)
    tr_w = x.corrwith(x.shift(5), axis=1)
    tr_m = x.corrwith(x.shift(20), axis=1)
    
    tr_r_d = x.corrwith(x.shift(), axis=1, method='spearman') ** 2
    tr_r_w = x.corrwith(x.shift(5), axis=1, method='spearman') ** 2
    tr_r_m = x.corrwith(x.shift(20), axis=1, method='spearman') ** 2
    
    df = DataFrame({'TR_D':tr_d,
                    'TR_W':tr_w,
                    'TR_M':tr_m,
                    'RANK_TR_D':tr_r_d,
                    'RANK_TR_W':tr_r_w,
                    'RANK_TR_M':tr_r_m,
                    })
    df = df.loc[df.index>=start_date, :]
    df.loc[:, 'white_threshold'] = '%s'%white_threshold
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.loc[:, 'FACTOR_NAME'] = factor_name
    df.loc[:, 'industry'] = 'WHITE'

    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")
    df.to_sql('tdailytr', engine, schema='factorevaluation', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

    
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
    # start_date = '20100101'
    
    factors = [
        'mc', 'bp', 
        'quality', 'value',
        'momentum', 'volatility', 'liquidity', 'corrmarket',
        'dailytech', 'hftech', 
        ]
    # factors = ['momentum', 'value']
    white_thresholds = [0, 0.2, 1-0.618, 0.618, 0.8]
    # white_thresholds = [0.9]
    pool = mp.Pool(4)
    for factor in factors:
        for white_threshold in white_thresholds:
            pool.apply_async(func=f, args=(factor, white_threshold, start_date, end_date))
    pool.close()
    pool.join()