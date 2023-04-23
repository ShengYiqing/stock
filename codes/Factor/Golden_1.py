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
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    try:
        sql = """
        CREATE TABLE `factor`.`tfactorgolden` (
          `REC_CREATE_TIME` VARCHAR(14) NULL,
          `TRADE_DATE` VARCHAR(8) NOT NULL,
          `STOCK_CODE` VARCHAR(18) NOT NULL,
          `FACTOR_VALUE` DOUBLE NULL,
          `PREPROCESSED_FACTOR_VALUE` DOUBLE NULL,
          `NEUTRAL_FACTOR_VALUE` DOUBLE NULL,
          PRIMARY KEY (`TRADE_DATE`, `STOCK_CODE`))
        """
        with engine.connect() as con:
            con.execute(sql)
    except:
        pass
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    sql = """
    select trade_date, stock_code from 
    label.tdailylabel 
    where trade_date >= {start_date}
    and trade_date <= {end_date}
    """.format(start_date=start_date, end_date=end_date)
    df = pd.read_sql(sql, engine)
    
    start_date_sql = tools.trade_date_shift(start_date, 250)
    start_month = start_date_sql[:6]
    end_month = end_date[:6]
    # end_month = (datetime.datetime.strptime(end_month, '%Y%m') + datetime.timedelta(days=32)).strftime('%Y%m')
    sql_f = """
    select stock_code, replace(adddate(concat(month, '01'), interval 1 month), '-', '') trade_date, count(1) times from tsdata.ttsbrokerrecommend
    where month >= {start_month}
    and month <= {end_month}
    group by stock_code, month
    """.format(start_month=start_month, end_month=end_month)
    df_f = pd.read_sql(sql_f, engine)
    f_m = df_f.set_index(['trade_date', 'stock_code']).times.unstack().fillna(0)
    f_d = df_f.set_index(['trade_date', 'stock_code']).times.unstack().fillna(0).diff()
    f_d[f_d<0] = 0
    f_m.index = pd.to_datetime(f_m.index)
    f_d.index = pd.to_datetime(f_d.index)
    f_m = f_m.resample('d').bfill()
    f_d = f_d.resample('d').bfill()
    f_m.index = [i.strftime('%Y%m%d') for i in f_m.index]
    f_d.index = [i.strftime('%Y%m%d') for i in f_d.index]
    f_m = Series(f_m.stack(), index=df.set_index(['trade_date', 'stock_code']).index).fillna(0)
    f_d = Series(f_d.stack(), index=df.set_index(['trade_date', 'stock_code']).index).fillna(0)
    f = f_m.groupby('trade_date').rank(pct=True) + f_d.groupby('trade_date').rank(pct=True)
    
    df = f.unstack()
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    df_p = tools.standardize(tools.winsorize(df))
    df_n = tools.neutralize(df)
    df_new = pd.concat([df, df_p, df_n], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE', 'NEUTRAL_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorgolden', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20200301'
    generate_factor(start_date, end_date)