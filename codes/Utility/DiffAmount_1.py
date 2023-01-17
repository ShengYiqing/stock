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
    start_date = tools.trade_date_shift(start_date, 250)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    
    sql = """
    select STOCK_CODE, TRADE_DATE, 
    (buy_sm_amount + buy_md_amount + buy_lg_amount + buy_elg_amount) BUY_AMOUNT,
    (sell_sm_amount + sell_md_amount + sell_lg_amount + sell_elg_amount) SELL_AMOUNT
    from ttsmoneyflow
    where trade_date >= {start_date}
    and trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date, end_date=end_date)
    df = pd.read_sql(sql, engine)
    BUY_AMOUNT = df.set_index(['TRADE_DATE', 'STOCK_CODE']).loc[:, 'BUY_AMOUNT']
    SELL_AMOUNT = df.set_index(['TRADE_DATE', 'STOCK_CODE']).loc[:, 'SELL_AMOUNT']
    BUY_AMOUNT = BUY_AMOUNT.unstack().replace(0, np.nan)
    SELL_AMOUNT = SELL_AMOUNT.unstack().replace(0, np.nan)
    a = np.log(BUY_AMOUNT) - np.log(SELL_AMOUNT)
    df_copy = a.copy()
    n_list = [5, 20, 60, 250]
    for n in n_list:
        df = df_copy.rolling(n, min_periods=int(np.ceil(n*4/5))).mean()
        df_p = tools.standardize(tools.winsorize(df))
        df_new = pd.concat([df, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
        df_new = df_new.stack()
        df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
        df_new.to_sql('tfactordiffamount%s'%n, engine, schema='factor', if_exists='append', index=True, chunksize=10000, dtype={'STOCK_CODE':VARCHAR(20), 'TRADE_DATE':VARCHAR(8), 'REC_CREATE_TIME':VARCHAR(14)}, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)