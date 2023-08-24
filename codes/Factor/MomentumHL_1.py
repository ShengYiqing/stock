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
    t1.high, t1.low, t1.close,
    t2.adj_factor 
    from ttsdaily t1
    left join ttsadjfactor t2
    on t1.stock_code = t2.stock_code
    and t1.trade_date = t2.trade_date
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

    hl = (np.log(h) - np.log(l)).unstack()
    
    w = hl - hl.rolling(60, min_periods=20).mean()

    trade_dates = tools.get_trade_cal(start_date, end_date)

    n = 250
    q_list = [0.05, 0.8]
        
    j = q_list[0]
    k = q_list[1]
    dic = {}
    for trade_date in trade_dates:
        print(trade_date, n, j, k)
        r_tmp = r.loc[r.index<=trade_date]
        w_tmp = w.loc[w.index<=trade_date]
        r_tmp = r_tmp.iloc[(-n):(-20)]
        w_tmp = w_tmp.iloc[(-n):(-20)]
        w_tmp = w_tmp.dropna(axis=1, thresh=0.618*n)
        w_tmp = w_tmp.rank(pct=True)
        w_tmp = ((w_tmp>=j)&(w_tmp<=k)).astype(int).replace(0, np.nan)
        dic[trade_date] = (r_tmp * w_tmp).mean().dropna()
    
    x = DataFrame(dic).T
    df = x
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactormomentumhl', engine, schema='factor', if_exists='append', index=True, chunksize=10000, dtype={'STOCK_CODE':VARCHAR(20), 'TRADE_DATE':VARCHAR(8), 'REC_CREATE_TIME':VARCHAR(14)}, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)