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
        CREATE TABLE `factor`.`tfactorgoldend` (
          `REC_CREATE_TIME` VARCHAR(14) NULL,
          `TRADE_DATE` VARCHAR(8) NOT NULL,
          `STOCK_CODE` VARCHAR(18) NOT NULL,
          `FACTOR_VALUE` DOUBLE NULL,
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
    f_d = df_f.set_index(['trade_date', 'stock_code']).times.unstack().apply(lambda x:x.sub(x.shift(), fill_value=0))
    f_d.replace(0, np.nan)
    f_d.index = pd.to_datetime(f_d.index)
    f_d = f_d.resample('d').bfill()
    f_d.index = [i.strftime('%Y%m%d') for i in f_d.index]
    f_d.index.name = 'trade_date'
    
    df = f_d
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    # df = tools.neutralize(df)
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorgoldend', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20200301'
    generate_factor(start_date, end_date)