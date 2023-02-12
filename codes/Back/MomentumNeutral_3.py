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
    
    f_list = ['mc', 'bp', 'quality']
    
    sql = ' select tsb.industry industry, tm.stock_code, tm.preprocessed_factor_value m '
    for f in f_list:
        sql = sql + ' , t{f}.preprocessed_factor_value {f} '.format(f=f)
    sql = sql + ' from factor.tfactormomentum tm '
    for f in f_list:
        sql = sql + """
        left join factor.tfactor{f} t{f} 
        on tm.stock_code = t{f}.stock_code 
        and tm.trade_date = t{f}.trade_date 
        """.format(f=f)
    sql = sql + """
    left join tsdata.ttsstockbasic tsb
    on tm.stock_code = tsb.stock_code
    """
    trade_cal = tools.get_trade_cal(start_date, end_date)
    
    factor_dic = {}
    for trade_date in trade_cal:
        # trade_date = '20221102'
        print(trade_date)
        sql_t = sql + ' where tmc.trade_date = %s '%trade_date
        df_sql = pd.read_sql(sql_t, engine).set_index('stock_code')
        df_sql.loc[:, 'industry'] = [gc.INDUSTRY_DIC_INV[ind] for ind in df_sql.industry]
        # df_sql.dropna(subset=['m'], inplace=True)
        X = pd.concat([pd.get_dummies(df_sql.industry), df_sql.loc[:, f_list]], axis=1).fillna(0)
        y = df_sql.loc[:, 'm']
        
        model = LinearRegression(n_jobs=-1)
        model.fit(X, y)
        y_predict = Series(model.predict(X), index=y.index)
        
        res = y - y_predict
        factor_dic[trade_date] = res
        
    factor = DataFrame(factor_dic).T
    
    factor = factor.replace(np.inf, np.nan)
    factor = factor.replace(-np.inf, np.nan)
    factor.index.name = 'trade_date'
    factor.columns.name = 'stock_code'
        
    factor_p = tools.standardize(tools.winsorize(factor))
    df_new = pd.concat([factor, factor_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactormomentumneutral', engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)