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
    trade_dates = tools.get_trade_cal(start_date, end_date)
    sql = """
    select stock_code, report_date trade_date, quarter, op_rt from ttsreportrc
    where report_date >= {start_date_sql}
    and report_date <= {end_date}
    """
    sql = sql.format(start_date_sql=start_date_sql, end_date=end_date)
    df_sql = pd.read_sql(sql, engine)
    
    
    
    sql_stock = """
    select stock_code from ttsstockbasic
    """
    stocks = pd.read_sql(sql_stock, engine).loc[:, 'stock_code']
    
    df = DataFrame(0, index=trade_dates, columns=stocks)
    df.index.name = 'TRADE_DATE'
    df.columns.name = 'STOCK_CODE'
    df.loc[:, :] = df_sql.groupby(['trade_date', 'stock_code']).count().loc[:, 'quarter'].unstack()
    
    df = df.fillna(0).rolling(250, min_periods=60).mean()
    df = df.loc[df.index>=start_date]
    df_p = tools.standardize(tools.winsorize(df))
    df_n = tools.neutralize(df)
    df_new = pd.concat([df, df_p, df_n], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE', 'NEUTRAL_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactoranalystcoverage', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)