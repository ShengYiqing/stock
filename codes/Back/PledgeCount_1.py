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
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    
    sql = """
    select STOCK_CODE, END_DATE TRADE_DATE, PLEDGE_COUNT from ttspledgestat
    where END_DATE >= {start_date}
    and END_DATE <= {end_date}
    """
    sql = sql.format(start_date=start_date, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['TRADE_DATE', 'STOCK_CODE'])
    PLEDGE_COUNT = df.loc[:, 'PLEDGE_COUNT']
    PLEDGE_COUNT = PLEDGE_COUNT.unstack()
    PLEDGE_COUNT = np.log(PLEDGE_COUNT)
    trade_dates = tools.get_trade_cal(start_date, end_date)
    PLEDGE_COUNT = DataFrame(PLEDGE_COUNT, index=trade_dates).fillna(method='ffill')
    PLEDGE_COUNT.index.name = 'TRADE_DATE'
    PLEDGE_COUNT.replace(np.inf, np.nan, inplace=True)
    PLEDGE_COUNT.replace(-np.inf, np.nan, inplace=True)
    df = PLEDGE_COUNT.copy()
    df_p = tools.standardize(tools.winsorize(df))
    df_n = tools.neutralize(df)
    df_new = pd.concat([df, df_p, df_n], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE', 'NEUTRAL_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorpledgecount', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)