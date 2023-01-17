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
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/mindata?charset=utf8")
    engine_w = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
        
    trade_cal = tools.get_trade_cal(start_date, end_date)
    for trade_date in trade_cal:
        
        sql = """
        select STOCK_CODE, TRADE_TIME, CLOSE from tmindata{TRADE_DATE}
        where trade_time >= '092500'
        and trade_time <= '150000'
        """.format(TRADE_DATE=trade_date)
        df_sql = pd.read_sql(sql, engine)
        CLOSE = df_sql.set_index(['TRADE_TIME', 'STOCK_CODE']).loc[:, 'CLOSE']
        CLOSE = CLOSE.unstack()
        CLOSE = np.log(CLOSE)
        r = CLOSE.diff()
        df = DataFrame(r.std(), columns=[trade_date]).T
        df.index.name='TRADE_DATE'
        df_p = tools.standardize(tools.winsorize(df))
        df_new = pd.concat([df, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
        df_new = df_new.stack()
        df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        df_new.to_sql('tfactorminsigma', engine_w, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(7)).strftime('%Y%m%d')
    start_date = '20210104'
    generate_factor(start_date, end_date)