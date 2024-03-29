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
    start_date_sql = tools.trade_date_shift(start_date, 250)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

    sql = """
    select t1.trade_date, t1.stock_code, 
    t1.close, t2.adj_factor, 
    tmf.buy_sm_vol, tmf.sell_sm_vol,  
    t1.vol
    from ttsdaily t1
    left join ttsadjfactor t2
    on t1.stock_code = t2.stock_code
    and t1.trade_date = t2.trade_date
    left join ttsmoneyflow tmf
    on t1.stock_code = tmf.stock_code
    and t1.trade_date = tmf.trade_date
    where t1.trade_date >= {start_date}
    and t1.trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date_sql, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    c = df.loc[:, 'close']
    af = df.loc[:, 'adj_factor']
    v = df.loc[:, 'vol']
    buy_sm_vol = df.loc[:, 'buy_sm_vol']
    sell_sm_vol = df.loc[:, 'sell_sm_vol']

    sm_vol = ((buy_sm_vol + sell_sm_vol) / v).unstack()
    sm_net = ((buy_sm_vol - sell_sm_vol) / v).unstack()

    r = np.log(c * af).unstack().diff()
    w = sm_net
    n = 20
    df = r.ewm(halflife=n).corr(w)
    # df = df * r.ewm(halflife=n).std()
    # df = df / w.ewm(halflife=n).std()
    df = df.replace(-np.inf, np.nan).replace(np.inf, np.nan)
    
    df = df.loc[df.index>=start_date]
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    # df = tools.neutralize(df)
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorcrsmnet', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)