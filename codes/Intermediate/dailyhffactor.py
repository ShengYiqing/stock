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

import Global_Config as gc
import tools
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR
import statsmodels.formula.api as smf

def Beta(df_sql):
    c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    c = c.loc[c.index>='093000']
    r = np.log(c).diff()
    r_m = r.mean(1)
    beta = r.corrwith(r_m) * r.std()
    df = DataFrame({'factor_value':beta})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'beta'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def CorrMarket(df_sql):
    c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    c = c.loc[c.index>='093000']
    r = np.log(c).diff()
    r_m = r.mean(1)
    corr = r.corrwith(r_m)
    df = DataFrame({'factor_value':corr})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'corrmarket'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def sigma(df_sql):
    c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    c = c.loc[c.index>='093000']
    r = np.log(c).diff()
    s = r.std()
    df = DataFrame({'factor_value':s})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'sigma'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def skew(df_sql):
    c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    c = c.loc[c.index>='093000']
    r = np.log(c).diff()
    s = r.skew()
    df = DataFrame({'factor_value':s})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'skew'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def HL(df_sql):
    h = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'high'].unstack()
    l = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'low'].unstack()
    h = h.loc[h.index>='093000']
    l = l.loc[l.index>='093000']
    hl = np.log(h) - np.log(l)
    hl = hl.mean()
    df = DataFrame({'factor_value':hl})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'hl'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def ETR(df_sql):
    v = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'vol'].unstack()
    v = v.loc[v.index>='093000']
    v = v.div(v.sum(), axis=1)
    v = v.replace(np.inf, np.nan)
    v = v.replace(-np.inf, np.nan)
    logv = np.log(v).replace(np.inf, np.nan).replace(-np.inf, np.nan)
    e = - (v * logv).sum()
    df = DataFrame({'factor_value':e})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'etr'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
    
def TTR(df_sql):
    v = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'vol'].unstack()
    v = v.loc[v.index>='093000']
    v = v.div(v.sum(), axis=1)
    v = v.replace(np.inf, np.nan)
    v = v.replace(-np.inf, np.nan)
    def g(s):
        s = s.dropna()
        if len(s) < 10:
            return np.nan
        t1 = np.arange(len(s))
        t2 = t1 ** 2
        data = DataFrame({'s':s,
                          't1':t1, 
                          't2':t2,})
        results = smf.ols('s ~ t1+t2', data=data).fit()
        return results.params.loc['t1']
    t = v.apply(g, axis=0)
    df = DataFrame({'factor_value':t})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'ttr'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
    
def UTR(df_sql):
    v = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'vol'].unstack()
    v = v.loc[v.index>='093000']
    v = v.div(v.sum(), axis=1)
    v = v.replace(np.inf, np.nan)
    v = v.replace(-np.inf, np.nan)
    def g(s):
        s = s.dropna()
        if len(s) < 10:
            return np.nan
        t1 = np.arange(len(s))
        t2 = t1 ** 2
        data = DataFrame({'s':s,
                          't1':t1, 
                          't2':t2,})
        results = smf.ols('s ~ t1+t2', data=data).fit()
        return results.params.loc['t2']
    u = v.apply(g, axis=0)
    df = DataFrame({'factor_value':u})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'utr'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
    
def spread(df_sql):
    s = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'spread'].unstack()
    s = s.loc[s.index>='093000']
    s = s.mean()
    df = DataFrame({'factor_value':s})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'spread'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
    
def imbalance(df_sql):
    i = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'imbalance'].unstack()
    i = i.loc[i.index>='093000']
    i = i.mean()
    df = DataFrame({'factor_value':i})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'imbalance'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
    
def PVCorr(df_sql):
    p = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    v = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'vol'].unstack()
    p = p.loc[p.index>='093000']
    v = v.loc[v.index>='093000']
    p = np.log(p).replace(-np.inf, np.nan)
    v = np.log(v).replace(-np.inf, np.nan)
    c = p.corrwith(v, method='spearman')
    df = DataFrame({'factor_value':c})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'pvcorr'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
       
def RVCorr(df_sql):
    p = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    v = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'vol'].unstack()
    p = p.loc[p.index>='093000']
    v = v.loc[v.index>='093000']
    p = np.log(p).replace(-np.inf, np.nan)
    r = p.diff()
    v = np.log(v).replace(-np.inf, np.nan)
    c = r.corrwith(v, method='spearman')
    df = DataFrame({'factor_value':c})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'rvcorr'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def PSCorr(df_sql): 
    p = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    s = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'spread'].unstack()
    p = p.loc[p.index>='093000']
    s = s.loc[s.index>='093000']
    p = np.log(p).replace(-np.inf, np.nan)
    s = np.log(s).replace(-np.inf, np.nan)
    c = p.corrwith(s, method='spearman')
    df = DataFrame({'factor_value':c})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'pscorr'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def PICorr(df_sql): 
    p = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    i = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'imbalance'].unstack()
    p = p.loc[p.index>='093000']
    i = i.loc[i.index>='093000']
    p = np.log(p).replace(-np.inf, np.nan)
    c = p.corrwith(i, method='spearman')
    df = DataFrame({'factor_value':c})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'picorr'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def intradaymomentum(df_sql): 
    c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    c = c.loc[c.index>='093000']
    r = np.log(c).diff()
    m = r.loc[r.index<='100000'].sum() - r.loc[r.index>'100000'].sum()
    df = DataFrame({'factor_value':m})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'intradaymomentum'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def callauctionmomentum(df_sql): 
    c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    c = c.loc[c.index<='092500']
    r = np.log(c).diff()
    m = r.loc[r.index>'092000'].sum() - r.loc[r.index<='092000'].sum()
    df = DataFrame({'factor_value':m})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'callauctionmomentum'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

def bluff(df_sql):
    h = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'high'].unstack()
    l = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'low'].unstack()
    h = h.loc[h.index<='092000']
    l = l.loc[l.index<='092000']
    h = h.max()
    l = l.min()
    c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
    c = c.loc['092500']
    hc = (np.log(h) - np.log(c))
    cl = (np.log(c) - np.log(l))
    b = hc - cl
    df = DataFrame({'factor_value':b})
    df.loc[:, 'trade_date'] = trade_date
    df.loc[:, 'factor_name'] = 'bluff'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

f_list = [Beta, CorrMarket, 
          sigma, skew, HL, 
          ETR, TTR, UTR, 
          spread, imbalance, 
          PVCorr, RVCorr, PSCorr, PICorr, 
          intradaymomentum, callauctionmomentum, 
          bluff]
# f_list = [Beta, PVCorr, RVCorr, PSCorr, PICorr, ]
#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')
    # start_date = '20210101'
    # end_date = '20230303'
    # start_date = '20230303'
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/mindata?charset=utf8")
    
    sql_tmp = """
    select trade_time, stock_code, open, high, low, close, spread, vol, amount, imbalance from tmindata{trade_date}
    where trade_date = {trade_date}
    """
    
    trade_dates = tools.get_trade_cal(start_date, end_date)
    
    for trade_date in trade_dates:
        print(trade_date)
        sql = sql_tmp.format(trade_date=trade_date)
        df_sql = pd.read_sql(sql, engine)
        for f in f_list:
            f(df_sql)