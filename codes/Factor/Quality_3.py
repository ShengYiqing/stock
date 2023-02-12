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
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tushare as ts
import itertools
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR

#%%
def generate_factor(start_date, end_date):
    
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    # 净资产
    # 经营资产
    # 营业收入
    # 核心利润
    # 经营现金流
    # 预期收益
    
    f_list = ['fjyzcl',
              'fyysrl', 'fyysrxjl', 
              'fmll', 'fmlxjl', 'fhxlrl', 'fzhsyl', 'fjyxjlll', 
              ]
    a_list = []
    for f in f_list:
        a_list.append(f+'d')
        a_list.append(f+'s')
    f_list.extend(a_list)
    
    sql = ' select tmc.trade_date, tmc.stock_code, tmc.factor_value mc '
    for f in f_list:
        sql = sql + ' , t{f}.factor_value {f} '.format(f=f)
    sql = sql + ' from factor.tfactormc tmc '
    for f in f_list:
        sql = sql + """
        left join factor.tfactor{f} t{f} 
        on tmc.stock_code = t{f}.stock_code 
        and tmc.trade_date = t{f}.trade_date 
        """.format(f=f)
    sql = sql + ' where tmc.trade_date >= %s '%start_date
    sql = sql + ' and tmc.trade_date <= %s '%end_date
        
    df_sql = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])

    df_sql.loc[:, f_list] = df_sql.loc[:, f_list].groupby('trade_date').apply(lambda x:x.rank()/x.notna().sum())
    df_sql.loc[:, [f for f in f_list if f[-1] == 'l']] = df_sql.loc[:, [f for f in f_list if f[-1] == 'l']] * 4
    df_sql.loc[:, [f for f in f_list if f[-1] == 'd']] = df_sql.loc[:, [f for f in f_list if f[-1] == 'd']] * 2
    df_sql.loc[:, [f for f in f_list if f[-1] == 's']] = df_sql.loc[:, [f for f in f_list if f[-1] == 's']] * (-1)
    df_sql.loc[:, 'q'] = df_sql.loc[:, f_list].mean(1)
    factor = df_sql.loc[:, 'q'].unstack()
    factor_p = tools.standardize(tools.winsorize(factor))
    df_new = pd.concat([factor, factor_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorquality', engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)