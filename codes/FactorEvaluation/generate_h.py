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
    
    sql = tools.generate_sql_y_x([factor_name], start_date, end_date, white_threshold, factor_value_type='factor_value')
    
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    df = df.loc[:, ['r_daily', 'r_weekly', 'r_monthly', factor_name]].dropna(subset=factor_name)
    
    y_d = tools.standardize(tools.winsorize(df.loc[:, 'r_daily'].unstack()))
    y_w = tools.standardize(tools.winsorize(df.loc[:, 'r_weekly'].unstack()))
    y_m = tools.standardize(tools.winsorize(df.loc[:, 'r_monthly'].unstack()))
    x = tools.standardize(tools.winsorize(df.loc[:, factor_name].unstack()))
    
    b_d = x.corrwith(y_d, axis=1)
    b_w = x.corrwith(y_w, axis=1)
    b_m = x.corrwith(y_m, axis=1)
    
    e2_d = (y_d - x.mul(b_d, axis=0)) ** 2
    e2_w = (y_w - x.mul(b_w, axis=0)) ** 2
    e2_m = (y_m - x.mul(b_m, axis=0)) ** 2
    
    h_d = x.corrwith(e2_d, axis=1, method='spearman') ** 2
    h_w = x.corrwith(e2_w, axis=1, method='spearman') ** 2
    h_m = x.corrwith(e2_m, axis=1, method='spearman') ** 2
    
    y_r_d = tools.standardize(df.loc[:, 'r_daily'].unstack().rank(1))
    y_r_w = tools.standardize(df.loc[:, 'r_weekly'].unstack().rank(1))
    y_r_m = tools.standardize(df.loc[:, 'r_monthly'].unstack().rank(1))
    x_r = tools.standardize(df.loc[:, factor_name].unstack().rank(1))
    
    b_r_d = x_r.corrwith(y_r_d, axis=1)
    b_r_w = x_r.corrwith(y_r_w, axis=1)
    b_r_m = x_r.corrwith(y_r_m, axis=1)
    
    e2_r_d = (y_r_d - x_r.mul(b_r_d, axis=0)) ** 2
    e2_r_w = (y_r_w - x_r.mul(b_r_w, axis=0)) ** 2
    e2_r_m = (y_r_m - x_r.mul(b_r_m, axis=0)) ** 2
    
    h_r_d = x_r.corrwith(e2_r_d, axis=1, method='spearman') ** 2
    h_r_w = x_r.corrwith(e2_r_w, axis=1, method='spearman') ** 2
    h_r_m = x_r.corrwith(e2_r_m, axis=1, method='spearman') ** 2
    
    df = DataFrame({'H_D':h_d,
                    'H_W':h_w,
                    'H_M':h_m,
                    'RANK_H_D':h_r_d,
                    'RANK_H_W':h_r_w,
                    'RANK_H_M':h_r_m,
                    })
    df.loc[:, 'white_threshold'] = '%s'%white_threshold
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.loc[:, 'FACTOR_NAME'] = factor_name
    df.loc[:, 'industry'] = 'WHITE'

    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")
    df.to_sql('tdailyh', engine, schema='factorevaluation', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

    
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
    # factors = ['momentum', 'value']
    white_thresholds = [0, 0.2, 1-0.618, 0.618, 0.8]
    # white_thresholds = [0.9]
    pool = mp.Pool(4)
    for factor in factors:
        for white_threshold in white_thresholds:
            pool.apply_async(func=f, args=(factor, white_threshold, start_date, end_date))
    pool.close()
    pool.join()