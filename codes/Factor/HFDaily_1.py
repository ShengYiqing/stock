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
    start_date_sql = tools.trade_date_shift(start_date, 20)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    
    sql = """
    select trade_date, stock_code, factor_name, factor_value from intermediate.tdailyhffactor
    where trade_date >= {start_date}
    and trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date_sql, end_date=end_date)
    df_sql = pd.read_sql(sql, engine)
    for factor_name in set(list(df_sql.factor_name)):
        try:
            sql = """
                CREATE TABLE `factor`.`tfactorhf%s` (
              `REC_CREATE_TIME` VARCHAR(14) NULL DEFAULT ' ',
              `STOCK_CODE` VARCHAR(20) NOT NULL DEFAULT ' ',
              `TRADE_DATE` VARCHAR(8) NOT NULL DEFAULT ' ',
              `FACTOR_VALUE` DOUBLE NULL,
              `PREPROCESSED_FACTOR_VALUE` DOUBLE NULL,
              `NEUTRAL_FACTOR_VALUE` DOUBLE NULL,
              PRIMARY KEY (`STOCK_CODE`, `TRADE_DATE`))
                """%factor_name
            with engine.connect() as con:
                con.execute(sql)
        except:
            pass
        df = df_sql.loc[df_sql.factor_name == factor_name]
        df = df.set_index(['trade_date', 'stock_code']).loc[:, 'factor_value'].unstack().sort_index()
        df = df.ewm(halflife=5).mean()
        df = df.loc[df.index>=start_date]
        df_p = tools.standardize(tools.winsorize(df))
        df_n = tools.neutralize(df)
        df_new = pd.concat([df, df_p, df_n], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE', 'NEUTRAL_FACTOR_VALUE'])
        df_new = df_new.stack()
        df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        df_new.to_sql('tfactorhf%s'%factor_name, engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(20)).strftime('%Y%m%d')
    start_date = '20210101'
    generate_factor(start_date, end_date)