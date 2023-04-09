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

def f(factor_name, df, start_date, end_date):
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
    df = df.loc[df.index>=start_date, :]
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.loc[:, 'FACTOR_NAME'] = factor_name

    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")
    df.to_sql('tdailyic', engine, schema='factorevaluation', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

    
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
    # start_date = '20100101'
    
    factors = [
        'operation', 'profitability', 'growth', 
        'momentum', 'volatility', 'liquidity', 'corrmarket',
        'dailytech', 'hftech', 
        ]
    neutral_list = ['operation', 'profitability', 'growth', ]
    
    factor_value_type_dic = {factor: 'neutral_factor_value' if factor in neutral_list else 'preprocessed_factor_value' for factor in factors}
    
    sql = tools.generate_sql_y_x(factors, start_date, end_date, factor_value_type_dic=factor_value_type_dic)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    pool = mp.Pool(4)
    for factor in factors:
        pool.apply_async(func=f, args=(factor, df.loc[:, ['r_daily', 'r_weekly', 'r_monthly', factor]], start_date, end_date))
    pool.close()
    pool.join()