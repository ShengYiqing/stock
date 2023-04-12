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
    start_date_sql = tools.trade_date_shift(start_date, 500)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    trade_dates = tools.get_trade_cal(start_date, end_date)
    sql = """
    select stock_code, ann_date, end_date, holder_name, hold_ratio from ttstop10holders
    where ann_date >= {start_date_sql}
    and ann_date <= {end_date}
    """
    sql = sql.format(start_date_sql=start_date_sql, end_date=end_date)
    df_sql = pd.read_sql(sql, engine)
    df = DataFrame(np.nan, index=trade_dates, columns=sorted(set(list(df_sql.stock_code))))
    df.index.name = 'TRADE_DATE'
    df.columns.name = 'STOCK_CODE'
    df_sql.set_index(['stock_code', 'holder_name', 'end_date'], inplace=True)
    for trade_date in trade_dates:
        df_sql_tmp = df_sql.loc[df_sql.ann_date<trade_date, :].loc[:, ['hold_ratio']]
        df.loc[trade_date, :] = df_sql_tmp.sort_index().groupby(['stock_code', 'holder_name']).last().reset_index().set_index('stock_code').loc[:, 'hold_ratio'].groupby('stock_code').max()
    df = np.log(df)
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df_p = tools.standardize(tools.winsorize(df))
    df_n = tools.neutralize(df)
    df_new = pd.concat([df, df_p, df_n], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE', 'NEUTRAL_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactortop1holders', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)