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
        CREATE TABLE `factor`.`tfactorcxx` (
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
    factor_dic = {
        'crhld': -1, 
        
        'crhl2c': -1,
        'crhl2o': 1,
        
        'crsmnet': 1, 
        'crsmvold': 1, 
        
        'crtrd': -1,
        }
    sql = tools.generate_sql_y_x(factor_dic.keys(), start_date, end_date, is_white=False, is_trade=False, is_industry=False)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

    df = pd.read_sql(sql, engine)
    df = df.set_index(['trade_date', 'stock_code']).loc[:, factor_dic.keys()]
    df = df.groupby('trade_date').rank(pct=True)
    for factor in factor_dic.keys():
        df.loc[:, factor] = df.loc[:, factor] * factor_dic[factor]
    df = df.mean(1)
    df = df.unstack()
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    # df = tools.neutralize(df)
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorcxx', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)