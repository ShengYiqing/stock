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
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306")

    sql = """
    select t1.trade_date, t1.stock_code, 
    t1.high, t1.low, t1.close, 
    t2.adj_factor , 
    tud.up_limit, tud.down_limit 
    from tsdata.ttsdaily t1
    left join tsdata.ttsadjfactor t2
    on t1.stock_code = t2.stock_code
    and t1.trade_date = t2.trade_date
    left join tsdata.ttsstklimit tud
    on t1.stock_code = tud.stock_code
    and t1.trade_date = tud.trade_date
    where t1.trade_date >= {start_date}
    and t1.trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date_sql, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    c = df.loc[:, 'close']
    h = df.loc[:, 'high']
    l = df.loc[:, 'low']
    af = df.loc[:, 'adj_factor']
    r = np.log(c * af).unstack().diff()

    h = df.loc[:, 'high']
    l = df.loc[:, 'low']
    u = df.loc[:, 'up_limit']
    d = df.loc[:, 'down_limit']

    ud = (u == h) | (d == l)
    ud = ud.unstack().fillna(False)
    r[ud] = np.nan

    # # tr = df.loc[:, 'turnover_rate'].unstack()

    # hl = (np.log(h) - np.log(l)).unstack()

    # w = hl
    # w_t = [0, 0.8]

    # trade_dates = tools.get_trade_cal(start_date, end_date)

    # n = 220
    # dic = {}
    # for trade_date in trade_dates:
    #     print(trade_date)
    #     r_tmp = r.loc[r.index<=trade_date].iloc[(-n):(-20)].dropna(axis=1, thresh=0.618*n)
    #     w_tmp = w.loc[r_tmp.index, r_tmp.columns].rank(pct=True)
    #     w_tmp = ((w_tmp>=w_t[0])&(w_tmp<=w_t[1])).astype(int)
    #     dic[trade_date] = (r_tmp * w_tmp).mean().dropna()

    df = r.rolling(200, min_periods=60).mean()
    df = df.loc[df.index>=start_date]
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    # df = tools.neutralize(df)
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactormomentum', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(7)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)