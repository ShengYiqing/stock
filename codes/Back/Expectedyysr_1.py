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
    start_date_sql = tools.trade_date_shift(start_date, 1000)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    trade_dates = tools.get_trade_cal(start_date, end_date)
    sql = """
    select stock_code, report_date, quarter, op_rt from ttsreportrc
    where report_date >= {start_date_sql}
    and report_date <= {end_date}
    """
    sql = sql.format(start_date_sql=start_date_sql, end_date=end_date)
    df_sql = pd.read_sql(sql, engine)
    df_sql.loc[:, 'year'] = [s[:4] if len(s) > 4 else '2000' for s in df_sql.quarter]
    df_sql = df_sql.loc[[i[-1]=='4' for i in df_sql.quarter],:]
    
    n = 3
    df = DataFrame(np.nan, index=trade_dates, columns=sorted(set(list(df_sql.stock_code))))
    df.index.name = 'TRADE_DATE'
    df.columns.name = 'STOCK_CODE'
    
    for trade_date in trade_dates:
        days_to_year_end = 30 * (12 - int(trade_date[4:6])) + (30 - int(trade_date[6:8]))
        days_to_year_end_percent = days_to_year_end / 360
        
        year_now = trade_date[:4]
        
        tmp = DataFrame(np.nan, index=list(range(n)), columns=df.columns)
        for i in range(n):
            year_target = str(int(year_now) + i)
            mask = (df_sql.report_date<=trade_date) & (df_sql.year==year_target)
            df_sql_tmp = df_sql.loc[mask, :]
            tmp.loc[i, :] = df_sql_tmp.set_index(['report_date', 'stock_code']).loc[:, 'op_rt'].groupby(['report_date', 'stock_code']).mean().unstack().dropna(axis=1, thresh=3).ewm(halflife=20).mean().iloc[-1, :]
        tmp.loc[0, :] = tmp.loc[0, :] * days_to_year_end_percent
        tmp.loc[n-1, :] = tmp.loc[n-1, :] * (1 - days_to_year_end_percent)
        df.loc[trade_date, :] = tmp.mean() * n / (n-1)
    df_p = tools.standardize(tools.winsorize(df))
    df_new = pd.concat([df, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorexpectedyysr', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)