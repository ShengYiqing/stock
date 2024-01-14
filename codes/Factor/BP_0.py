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
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    
    sql = """
    select stock_code, trade_date, pb from ttsdailybasic
    where trade_date >= {start_date}
    and trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date, end_date=end_date)
    df = pd.read_sql(sql, engine)
    df = df.set_index(['trade_date', 'stock_code']).loc[:, 'pb']
    df = df.unstack()
    df = 1 / df
    df = np.log(df)
    df = df.loc[df.index>=start_date]
    df.replace(-np.inf, np.nan, inplace=True)
    df.replace(np.inf, np.nan, inplace=True)
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorbp', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)