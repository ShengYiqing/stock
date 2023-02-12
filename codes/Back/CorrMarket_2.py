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
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    
    sql = """
    select t5.STOCK_CODE, t5.TRADE_DATE, 
           t5.preprocessed_factor_value as c5, 
           t20.preprocessed_factor_value as c20,
           t60.preprocessed_factor_value as c60, 
           t250.preprocessed_factor_value as c250 
           from tfactorcorrmarket5 t5
           left join tfactorcorrmarket20 t20
           on t5.trade_date = t20.trade_date
           and t5.stock_code = t20.stock_code
           left join tfactorcorrmarket60 t60
           on t5.trade_date = t60.trade_date
           and t5.stock_code = t60.stock_code
           left join tfactorcorrmarket250 t250
           on t5.trade_date = t250.trade_date
           and t5.stock_code = t250.stock_code
    where t5.trade_date >= {start_date}
    and t5.trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['TRADE_DATE', 'STOCK_CODE'])
    df = df.mean(1)
    df = df.unstack()
    df_p = tools.standardize(tools.winsorize(df))
    df_new = pd.concat([df, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorcorrmarket', engine, schema='factor', if_exists='append', index=True, chunksize=10000, dtype={'STOCK_CODE':VARCHAR(20), 'TRADE_DATE':VARCHAR(8), 'REC_CREATE_TIME':VARCHAR(14)}, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)