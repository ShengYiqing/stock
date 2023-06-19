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
    select t1.STOCK_CODE, t1.TRADE_DATE, t1.CLOSE, t1.AMOUNT, t2.ADJ_FACTOR from ttsdaily t1
    left join ttsadjfactor t2
    on t1.STOCK_CODE = t2.STOCK_CODE
    and t1.TRADE_DATE = t2.TRADE_DATE
    where t1.trade_date >= {start_date}
    and t1.trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date_sql, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['TRADE_DATE', 'STOCK_CODE'])
    AMOUNT = df.loc[:, 'AMOUNT']
    CLOSE = df.loc[:, 'CLOSE']
    ADJ_FACTOR = df.loc[:, 'ADJ_FACTOR']
    AMOUNT = AMOUNT.unstack()
    CLOSE = CLOSE.unstack()
    ADJ_FACTOR = ADJ_FACTOR.unstack()
    CLOSE = CLOSE * ADJ_FACTOR
    VOL = AMOUNT / CLOSE
    VOL.replace(0, np.nan, inplace=True)
    VOL = np.log(VOL)
    CLOSE = np.log(CLOSE)
    r = CLOSE.diff()
    df = r.ewm(halflife=5).corr(VOL.shift())
    df = df.loc[df.index>=start_date]
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    # df = tools.neutralize(df)
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorcrsv', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)