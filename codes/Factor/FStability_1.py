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
    start_date_sql = tools.trade_date_shift(start_date, 750)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    sql = """
    select ann_date, end_date, ann_type, stock_code, financial_index, financial_value 
    from findata.tfindata
    where end_date >= {start_date}
    and end_date <= {end_date}
    and substr(end_date, -4) in ('0331', '0630', '0930', '1231')
    and ann_type in ('定期报告', '业绩预告', '业绩快报')
    and financial_index in ('yysr', 'gmjlr', 'jyxjll', 'zzc')
    """.format(start_date=start_date_sql, end_date=end_date)
    df_sql = pd.read_sql(sql, engine).sort_values(['financial_index', 'end_date', 'stock_code', 'ann_date'])

    trade_dates = tools.get_trade_cal(start_date, end_date)
    
    factor = DataFrame()
    dic = {}
    for trade_date in trade_dates:
        print(trade_date)
        start_date_tmp = tools.trade_date_shift(trade_date, 750)
        df_tmp = df_sql.loc[(df_sql.ann_date>=start_date_tmp)&(df_sql.ann_date<=trade_date)]
        df_tmp = df_tmp.groupby(['financial_index', 'end_date', 'stock_code']).last()
        
        yysr = df_tmp.loc['yysr'].loc[:, 'financial_value'].unstack()
        cols = yysr.columns
        yysr['YYYY'] = [ind[:4] for ind in yysr.index]
        yysr = yysr.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        yysr = yysr.loc[:, cols]
        
        gmjlr = df_tmp.loc['gmjlr'].loc[:, 'financial_value'].unstack()
        cols = gmjlr.columns
        gmjlr['YYYY'] = [ind[:4] for ind in gmjlr.index]
        gmjlr = gmjlr.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        gmjlr = gmjlr.loc[:, cols]
        
        jyxjll = df_tmp.loc['jyxjll'].loc[:, 'financial_value'].unstack()
        cols = jyxjll.columns
        jyxjll['YYYY'] = [ind[:4] for ind in jyxjll.index]
        jyxjll = jyxjll.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        jyxjll = jyxjll.loc[:, cols]
        
        zzc = df_tmp.loc['zzc'].loc[:, 'financial_value'].unstack()
        zzc[zzc<=0] = np.nan
        zzc = zzc.rolling(2, min_periods=1).mean()
        
        yysr_ttm = yysr.rolling(4, min_periods=1).mean()
        gmjlr_ttm = gmjlr.rolling(4, min_periods=1).mean()
        jyxjll_ttm = jyxjll.rolling(4, min_periods=1).mean()
        zzc_ttm = zzc.rolling(4, min_periods=1).mean()
        
        s_1 = (yysr_ttm / zzc_ttm).rolling(4).std().fillna(method='ffill', limit=4).iloc[-1].rank(ascending=False, pct=True)
        s_2 = (gmjlr_ttm / zzc_ttm).rolling(4).std().fillna(method='ffill', limit=4).iloc[-1].rank(ascending=False, pct=True)
        s_3 = (jyxjll_ttm / zzc_ttm).rolling(4).std().fillna(method='ffill', limit=4).iloc[-1].rank(ascending=False, pct=True)
        s = s_1 + s_2 + s_3
        
        dic[trade_date] = s
    factor = DataFrame(dic).T
    factor.index.name = 'trade_date'
    factor.columns.name = 'stock_code'
    # factor = tools.neutralize(factor)
    df = DataFrame({'factor_value':factor.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorstability', engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)