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
    select ann_date, end_date, ann_type, stock_code, financial_index, financial_value 
    from findata.tfindata
    where ann_date >= {start_date}
    and ann_date <= {end_date}
    and substr(end_date, -4) in ('0331', '0630', '0930', '1231')
    and ann_type in ('分析师预期', '定期报告', '业绩预告', '业绩快报')
    and financial_index in ('gmjlr', 'jzc')
    """.format(start_date=start_date_sql, end_date=end_date)
    df_sql = pd.read_sql(sql, engine).sort_values(['financial_index', 'end_date', 'stock_code', 'ann_date'])
    
    trade_dates = tools.get_trade_cal(start_date, end_date)
    
    factor = DataFrame()
    dic = {}
    for trade_date in trade_dates:
        print(trade_date)
        end_date_tmp = trade_date[:4] + '1231'
        start_date_tmp = tools.trade_date_shift(trade_date, 250)
        df_tmp = df_sql.loc[(df_sql.ann_type=='分析师预期')&(df_sql.financial_index=='gmjlr')&(df_sql.ann_date>=start_date_tmp)&(df_sql.ann_date<=trade_date)&(df_sql.end_date==end_date_tmp)]
        con_gmjlr = df_tmp.set_index(['ann_date', 'stock_code']).financial_value.unstack().ewm(halflife=60, min_periods=5).mean().iloc[-1]
        
        df_tmp = df_sql.loc[(df_sql.ann_type!='分析师预期')&(df_sql.financial_index=='jzc')&(df_sql.ann_date>=start_date_tmp)&(df_sql.ann_date<=trade_date)]
        jzc = df_tmp.set_index(['stock_code', 'end_date', 'ann_date']).financial_value.sort_index().groupby('stock_code').last()
        jzc[jzc<=0] = np.nan
        
        con_operation = (con_gmjlr / jzc).dropna()
        dic[trade_date] = con_operation
    factor = DataFrame(dic).T
    factor.index.name = 'trade_date'
    factor.columns.name = 'stock_code'
    # factor = tools.neutralize(factor)
    df = DataFrame({'factor_value':factor.stack().replace(-np.inf, np.nan).replace(np.inf, np.nan).dropna()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorconprofitability', engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)