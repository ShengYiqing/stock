#!/usr/bin/env python
# coding: utf-8

#%%
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import tushare as ts

import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR

#%%
def generate_factor(start_date, end_date):
    start_date = tools.trade_date_shift(start_date, 20)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/mindata?charset=utf8")
    
    sql = """
    select trade_date, stock_code, trade_time, close, vol from tmindata
    where trade_date >= {start_date}
    and trade_date <= {end_date}
    """.format(start_date=start_date, end_date=end_date)
    
    df_sql = pd.read_sql(sql, engine)
    def f(df):
        p = df.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
        v = df.set_index(['trade_time', 'stock_code']).loc[:, 'vol'].unstack()
        p = p.loc[p.index>='093000']
        v = v.loc[v.index>='093000']
        p = np.log(p).replace(-np.inf, np.nan)
        r = p.diff()
        v = np.log(v).replace(-np.inf, np.nan)
        
        c = r.corrwith(v)
        return c
    df = df_sql.groupby('trade_date').apply(f).unstack()
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df = df.ewm(halflife=5).mean()
    
    df_p = tools.standardize(tools.winsorize(df))
    df_new = pd.concat([df, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df_new.to_sql('tfactorminrvcorr', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(5)).strftime('%Y%m%d')
    start_date = '20210104'
    generate_factor(start_date, end_date)