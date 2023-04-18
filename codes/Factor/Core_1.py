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
    and financial_index in ('yysr', 'yycb', 'sjjfj', 'xsfy', 'glfy', 'cwfy', 'zzc')
    """.format(start_date=start_date_sql, end_date=end_date)
    df_sql = pd.read_sql(sql, engine).sort_values(['financial_index', 'end_date', 'stock_code', 'ann_date'])

    trade_dates = tools.get_trade_cal(start_date, end_date)
    
    factor = DataFrame()
    dic = {}
    for trade_date in trade_dates:
        print(trade_date)
        df_tmp = df_sql.loc[df_sql.ann_date<=trade_date]
        df_tmp = df_tmp.groupby(['financial_index', 'end_date', 'stock_code']).last()
        
        yysr = df_tmp.loc['yysr'].loc[:, 'financial_value'].unstack()
        cols = yysr.columns
        yysr['YYYY'] = [ind[:4] for ind in yysr.index]
        yysr = yysr.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        yysr = yysr.loc[:, cols]
        
        yycb = df_tmp.loc['yycb'].loc[:, 'financial_value'].unstack()
        cols = yycb.columns
        yycb['YYYY'] = [ind[:4] for ind in yycb.index]
        yycb = yycb.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        yycb = yycb.loc[:, cols]
        
        sjjfj = df_tmp.loc['sjjfj'].loc[:, 'financial_value'].unstack()
        cols = sjjfj.columns
        sjjfj['YYYY'] = [ind[:4] for ind in sjjfj.index]
        sjjfj = sjjfj.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        sjjfj = sjjfj.loc[:, cols]
        
        xsfy = df_tmp.loc['xsfy'].loc[:, 'financial_value'].unstack()
        cols = xsfy.columns
        xsfy['YYYY'] = [ind[:4] for ind in xsfy.index]
        xsfy = xsfy.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        xsfy = xsfy.loc[:, cols]
        
        glfy = df_tmp.loc['glfy'].loc[:, 'financial_value'].unstack()
        cols = glfy.columns
        glfy['YYYY'] = [ind[:4] for ind in glfy.index]
        glfy = glfy.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        glfy = glfy.loc[:, cols]
        
        cwfy = df_tmp.loc['cwfy'].loc[:, 'financial_value'].unstack()
        cols = cwfy.columns
        cwfy['YYYY'] = [ind[:4] for ind in cwfy.index]
        cwfy = cwfy.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        cwfy = cwfy.loc[:, cols]
        
        zzc = df_tmp.loc['zzc'].loc[:, 'financial_value'].unstack()
        zzc[zzc<=0] = np.nan
        zzc = zzc.rolling(2, min_periods=1).mean()
        
        yysr_ttm = yysr.rolling(4, min_periods=1).mean()
        yycb_ttm = yycb.rolling(4, min_periods=1).mean()
        sjjfj_ttm = sjjfj.rolling(4, min_periods=1).mean()
        xsfy_ttm = xsfy.rolling(4, min_periods=1).mean()
        glfy_ttm = glfy.rolling(4, min_periods=1).mean()
        cwfy_ttm = cwfy.rolling(4, min_periods=1).mean()
        zzc_ttm = zzc.rolling(4, min_periods=1).mean()
        
        core = (yysr_ttm - yycb_ttm - sjjfj_ttm - xsfy_ttm - glfy_ttm - cwfy_ttm) / zzc_ttm
        core = core.replace(-np.inf, np.nan).replace(np.inf, np.nan)
        core.fillna(method='ffill', limit=4, inplace=True)
        dic[trade_date] = core.iloc[-1]
    factor = DataFrame(dic).T
    factor.index.name = 'trade_date'
    factor.columns.name = 'stock_code'
    
    factor_p = tools.standardize(tools.winsorize(factor))
    factor_n = tools.neutralize(factor)
    df_new = pd.concat([factor, factor_p, factor_n], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE', 'NEUTRAL_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorcore', engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)