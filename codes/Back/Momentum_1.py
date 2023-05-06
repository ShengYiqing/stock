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
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    
    sql = """
    select 
    t1.trade_date, t1.stock_code, t1.close, t2.adj_factor, t3.l1_name, log(t4.total_mv) mc, 1 / t4.pb bp
    from tsdata.ttsdaily t1
    left join tsdata.ttsadjfactor t2
    on t1.STOCK_CODE = t2.STOCK_CODE
    and t1.TRADE_DATE = t2.TRADE_DATE
    left join indsw.tindsw t3
    on t1.STOCK_CODE = t3.STOCK_CODE
    left join tsdata.ttsdailybasic t4
    on t1.STOCK_CODE = t4.STOCK_CODE
    and t1.TRADE_DATE = t4.TRADE_DATE
    where t1.trade_date >= {start_date}
    and t1.trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date_sql, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    df = df.unstack().stack()
    df.loc[:, 'close'] = np.log(df.close * df.adj_factor)
    df.loc[:, 'r'] = df.close.groupby('stock_code').apply(lambda x:x.diff().ewm(halflife=20).mean())
    
    def f(df):
        data = df.loc[:, ['r', 'l1_name', 'mc', 'bp']].dropna()
        data.loc[:, 'r'] = tools.standardize(tools.winsorize(data.loc[:, 'r']))
        data.loc[:, 'mc'] = tools.standardize(tools.winsorize(data.loc[:, 'mc']))
        data.loc[:, 'bp'] = tools.standardize(tools.winsorize(data.loc[:, 'bp']))
        X = pd.concat([pd.get_dummies(data.loc[:, 'l1_name']), data.loc[:, ['mc', 'bp']]], axis=1).fillna(0)
    
        y = data.loc[:, 'r']
        
        y_predict = X.dot(np.linalg.inv(X.T.dot(X)+0.001*np.identity(len(X.T))).dot(X.T).dot(y))
        return tools.standardize(tools.winsorize(y_predict))
    
    df = df.groupby('trade_date').apply(f).reset_index(0, drop=True).unstack()
    
    df = df.loc[df.index>=start_date]
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.index.name = 'trade_date'
    df.columns.name = 'stock_code'
    df = DataFrame({'factor_value':df.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactormomentum', engine, schema='factor', if_exists='append', index=True, chunksize=10000, dtype={'STOCK_CODE':VARCHAR(20), 'TRADE_DATE':VARCHAR(8), 'REC_CREATE_TIME':VARCHAR(14)}, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)