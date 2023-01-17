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
    start_date = tools.trade_date_shift(start_date, 250)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    n = 5
    sql = """
    select STOCK_CODE, TRADE_DATE, FACTOR_VALUE from tfactorminsigma
    where TRADE_DATE >= {start_date}
    and TRADE_DATE <= {end_date}
    """.format(start_date=start_date, end_date=end_date)
    df_sql = pd.read_sql(sql, engine)
    FACTOR_VALUE = df_sql.set_index(['TRADE_DATE', 'STOCK_CODE']).loc[:, 'FACTOR_VALUE']
    FACTOR_VALUE = FACTOR_VALUE.unstack()
    
    df = FACTOR_VALUE.ewm(halflife=n).mean()
    df.index.name = 'TRADE_DATE'
    df_p = tools.standardize(tools.winsorize(df))
    df = pd.concat([df, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df = df.stack()
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tfactorminsigmamean', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(2)).strftime('%Y%m%d')
    start_date = '20210104'
    generate_factor(start_date, end_date)