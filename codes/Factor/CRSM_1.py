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
    start_date_sql = tools.trade_date_shift(start_date, 60)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

    sql = """
    select t1.trade_date, t1.stock_code, 
    t1.close, tmf.buy_sm_vol, tmf.sell_sm_vol,  
    t1.vol, 
    t2.adj_factor from tsdata.ttsdaily t1
    left join tsdata.ttsmoneyflow tmf
    on t1.stock_code = tmf.stock_code
    and t1.trade_date = tmf.trade_date
    left join tsdata.ttsadjfactor t2
    on t1.stock_code = t2.stock_code
    and t1.trade_date = t2.trade_date
    where t1.trade_date >= {start_date}
    and t1.trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date_sql, end_date=end_date)
    df = pd.read_sql(sql, engine)

    v = df.set_index(['trade_date', 'stock_code']).loc[:, 'vol']
    buy_sm_vol = df.set_index(['trade_date', 'stock_code']).loc[:, 'buy_sm_vol']
    sell_sm_vol = df.set_index(['trade_date', 'stock_code']).loc[:, 'sell_sm_vol']

    sm_vol_rate = (buy_sm_vol + sell_sm_vol) / v

    c = df.set_index(['trade_date', 'stock_code']).loc[:, 'close']

    adj_factor = df.set_index(['trade_date', 'stock_code']).loc[:, 'adj_factor']

    r = np.log(c * adj_factor).groupby('stock_code').diff()
    r = r.unstack()
    s = sm_vol_rate.unstack().rank(axis=1, pct=True)
    df = r.ewm(halflife=5).corr(s)
    df = df.loc[df.index>=start_date]
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    # df = tools.neutralize(df)
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorcrsm', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)