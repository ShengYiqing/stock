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
    n = 5
    start_date_sql = tools.trade_date_shift(start_date, n)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    sql = """
    select t1.trade_date, t1.stock_code, 
    t1.open, t1.high, t1.low, t1.close, 
    t2.adj_factor
    from tsdata.ttsdaily t1
    left join tsdata.ttsadjfactor t2
    on t1.stock_code = t2.stock_code
    and t1.trade_date = t2.trade_date
    where t1.trade_date >= {start_date}
    and t1.trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date_sql, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])

    c = df.loc[:, 'close']
    o = df.loc[:, 'open']
    adj_factor = df.loc[:, 'adj_factor']
    c = np.log(c * adj_factor).unstack()
    o = np.log(o * adj_factor).unstack()

    # r = c.diff()
    r = c - o
    df = r.ewm(halflife=1).mean()
    df = df.loc[df.index>=start_date]
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    # df = tools.neutralize(df)
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorreversal', engine, schema='factor', if_exists='append', index=True, chunksize=10000, dtype={'STOCK_CODE':VARCHAR(20), 'TRADE_DATE':VARCHAR(8), 'REC_CREATE_TIME':VARCHAR(14)}, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)