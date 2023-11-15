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
    y_d = tools.standardize(tools.winsorize(df.loc[:, 'r_d'].unstack()))
    x = tools.standardize(tools.winsorize(df.loc[:, factor_name].unstack()))
    
    ic_d = x.corrwith(y_d, axis=1)
    
    rank_ic_d = x.corrwith(y_d, axis=1, method='spearman')
    
    df_ic = DataFrame({'IC_D':ic_d, 'RANK_IC_D':rank_ic_d, })
    df_ic = df_ic.loc[df_ic.index>=start_date, :]
    df_ic.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df_ic.loc[:, 'FACTOR_NAME'] = factor_name

    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")
    df_ic.to_sql('tdailyic', engine, schema='factorevaluation', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)


    # e2_d = (y_d - x.mul(ic_d, axis=0)) ** 2
    
    # h_d = x.corrwith(e2_d, axis=1, method='spearman') ** 2
    
    # y_r_d = tools.standardize(y_d.rank(1))
    # x_r = tools.standardize(x.rank(1))
    
    # ic_r_d = x_r.corrwith(y_r_d, axis=1)
    
    # e2_r_d = (y_r_d - x_r.mul(ic_r_d, axis=0)) ** 2
    
    # h_r_d = x_r.corrwith(e2_r_d, axis=1, method='spearman') ** 2
    
    # df_h = DataFrame({'H_D':h_d, 'RANK_H_D':h_r_d, })
    # df_h = df_h.loc[df_h.index>=start_date, :]
    # df_h.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    # df_h.loc[:, 'FACTOR_NAME'] = factor_name

    # engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")
    # df_h.to_sql('tdailyh', engine, schema='factorevaluation', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

    
    # tr_d = x.corrwith(x.shift(), axis=1)
    
    # tr_r_d = x.corrwith(x.shift(), axis=1, method='spearman')
    
    # df_tr = DataFrame({'TR_D':tr_d, 'RANK_TR_D':tr_r_d, })
    # df_tr = df_tr.loc[df_tr.index>=start_date, :]
    # df_tr.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    # df_tr.loc[:, 'FACTOR_NAME'] = factor_name

    # engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")
    # df_tr.to_sql('tdailytr', engine, schema='factorevaluation', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)



if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
    start_date = '20100101'
    
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

    
    sql = tools.generate_sql_y_x(factors, start_date, end_date, label_type='c')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    pool = mp.Pool(4)
    for factor in factors:
        pool.apply_async(func=f, args=(factor, df.loc[:, ['r_d', factor]], start_date, end_date))
    pool.close()
    pool.join()