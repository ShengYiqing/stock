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
        CREATE TABLE `factor`.`tfactorquality` (
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
    factor_dic = {'operation':1, 
                  'gross':1, 
                  'core':1, 
                  'profitability':1, 
                  'cash':1, 
                  'growth':1,
                  'stability':1,
                  }
    factor_value_type_dic = {factor:'factor_value' for factor in factor_dic.keys()}
    sql = tools.generate_sql_y_x(factor_dic.keys(), start_date, end_date, is_white=False, is_trade=False, is_industry=False, factor_value_type_dic=factor_value_type_dic)
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
    df_p = tools.standardize(tools.winsorize(df))
    # df_n = tools.neutralize(df)
    # df_new = pd.concat([df, df_p, df_n], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE', 'NEUTRAL_FACTOR_VALUE'])
    df_new = pd.concat([df, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorquality', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)