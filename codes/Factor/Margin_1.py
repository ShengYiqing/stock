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
    select STOCK_CODE, TRADE_DATE, RZRQYE from ttsmargindetail 
    where trade_date >= {start_date_sql}
    and trade_date <= {end_date}
    """
    sql = sql.format(start_date_sql=start_date_sql, end_date=end_date)
    df = pd.read_sql(sql, engine)
    RZRQYE = df.set_index(['TRADE_DATE', 'STOCK_CODE']).loc[:, 'RZRQYE']
    
    trade_dates = tools.get_trade_cal(start_date_sql, end_date)
    RZRQYE = RZRQYE.unstack().shift()
    RZRQYE = DataFrame(RZRQYE, index=trade_dates).fillna(method='ffill')
    RZRQYE.index.name = 'TRADE_DATE'
    RZRQYE[RZRQYE<=0] = np.nan
    df_copy = np.log(RZRQYE)
    trade_dates = tools.get_trade_cal(start_date, end_date)
    n = 5
    df = df_copy.diff().ewm(halflife=n).mean()
    df = DataFrame(df, index=trade_dates)
    df.index.name = 'TRADE_DATE'
    df_p = tools.standardize(tools.winsorize(df))
    df_new = pd.concat([df, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactormargin', engine, schema='factor', if_exists='append', index=True, chunksize=10000, dtype={'STOCK_CODE':VARCHAR(20), 'TRADE_DATE':VARCHAR(8), 'REC_CREATE_TIME':VARCHAR(14)}, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(150)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)