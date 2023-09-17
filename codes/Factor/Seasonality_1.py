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
    start_date_sql = tools.trade_date_shift(start_date, 1250)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

    sql = """
    select t1.trade_date, t1.stock_code, 
    t1.close, t1.high, t1.low, t2.adj_factor, 
    tud.up_limit, tud.down_limit 
    from ttsdaily t1
    left join ttsadjfactor t2
    on t1.stock_code = t2.stock_code
    and t1.trade_date = t2.trade_date
    left join ttsstklimit tud
    on t1.stock_code = tud.stock_code
    and t1.trade_date = tud.trade_date
    where t1.trade_date >= {start_date}
    and t1.trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date_sql, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    c = df.loc[:, 'close']
    af = df.loc[:, 'adj_factor']
    r = np.log(c * af).unstack().diff()

    h = df.loc[:, 'high']
    l = df.loc[:, 'low']
    u = df.loc[:, 'up_limit']
    d = df.loc[:, 'down_limit']

    ud = (u == h) | (d == l)
    ud = ud.unstack().fillna(False)
    r[ud] = np.nan
    dic = {}
    for i in range(1, 6):
        dic[i] = (6-i) * r.rolling(20, min_periods=5).mean().shift(240*i-20).stack()
    seasonality = DataFrame(dic).mean(1).unstack()
    df = seasonality.loc[seasonality.index>=start_date]
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    # df = tools.neutralize(df)
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorseasonality', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(7)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)