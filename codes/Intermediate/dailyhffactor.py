# #!/usr/bin/env python
# # coding: utf-8

# #%%
# import os
# import sys
# import time
# import datetime
# import numpy as np
# import pandas as pd
# from pandas import Series, DataFrame
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# import tushare as ts

# import Global_Config as gc
# import tools
# from sqlalchemy import create_engine
# from sqlalchemy.types import VARCHAR
# import statsmodels.formula.api as smf

# def Beta(df_sql):
#     c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
#     c = c.loc[c.index>='093000']
#     r = np.log(c).diff()
    
#     a = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'amount'].unstack().sum()
#     r_m = ((r*a) / a.sum()).sum(1)
#     beta = r.corrwith(r_m) * r.std() / r_m.std()
#     beta = beta.replace(np.inf, np.nan)
#     beta = beta.replace(-np.inf, np.nan)
#     df = DataFrame({'factor_value':beta})
#     df.loc[:, 'trade_date'] = trade_date
#     df.loc[:, 'factor_name'] = 'beta'
#     df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#     df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

# def Corrmarket(df_sql):
#     c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
#     c = c.loc[c.index>='093000']
#     r = np.log(c).diff()
    
#     a = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'amount'].unstack().sum()
#     r_m = ((r*a) / a.sum()).sum(1)
#     corrmarket = r.corrwith(r_m, method='spearman')
#     corrmarket = corrmarket.replace(np.inf, np.nan)
#     corrmarket = corrmarket.replace(-np.inf, np.nan)
#     df = DataFrame({'factor_value':corrmarket})
#     df.loc[:, 'trade_date'] = trade_date
#     df.loc[:, 'factor_name'] = 'corrmarket'
#     df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#     df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

# def sigma(df_sql):
#     c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
#     c = c.loc[c.index>='093000']
#     r = np.log(c).diff()
#     s = r.std()
#     df = DataFrame({'factor_value':s})
#     df.loc[:, 'trade_date'] = trade_date
#     df.loc[:, 'factor_name'] = 'sigma'
#     df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#     df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

# def sigmaopen(df_sql):
#     c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
#     c = c.loc[c.index>='091500']
#     c = c.loc[c.index<='092500']
#     r = np.log(c).diff()
#     s = r.std()
#     df = DataFrame({'factor_value':s})
#     df.loc[:, 'trade_date'] = trade_date
#     df.loc[:, 'factor_name'] = 'sigmaopen'
#     df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#     df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

# def skew(df_sql):
#     c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
#     c = c.loc[c.index>='093000']
#     r = np.log(c).diff()
#     s = r.skew()
#     df = DataFrame({'factor_value':s})
#     df.loc[:, 'trade_date'] = trade_date
#     df.loc[:, 'factor_name'] = 'skew'
#     df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#     df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

# def ETR(df_sql):
#     v = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'vol'].unstack()
#     v = v.loc[v.index>='093000']
#     v = v.div(v.sum(), axis=1)
#     v = v.replace(np.inf, np.nan)
#     v = v.replace(-np.inf, np.nan)
#     logv = np.log(v).replace(np.inf, np.nan).replace(-np.inf, np.nan)
#     e = - (v * logv).sum()
#     df = DataFrame({'factor_value':e})
#     df.loc[:, 'trade_date'] = trade_date
#     df.loc[:, 'factor_name'] = 'etr'
#     df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#     df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
    
# def spread(df_sql):
#     s = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'spread'].unstack()
#     s = s.loc[s.index>='093000']
#     s = s.mean()
#     df = DataFrame({'factor_value':s})
#     df.loc[:, 'trade_date'] = trade_date
#     df.loc[:, 'factor_name'] = 'spread'
#     df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#     df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
    
# def imbalance(df_sql):
#     i = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'imbalance'].unstack()
#     i = i.loc[i.index>='093000']
#     i = i.mean()
#     df = DataFrame({'factor_value':i})
#     df.loc[:, 'trade_date'] = trade_date
#     df.loc[:, 'factor_name'] = 'imbalance'
#     df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#     df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
    
# def momentum(df_sql):
#     c = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'close'].unstack()
#     c = c.loc[c.index>='093000']
#     r = np.log(c).diff()
    
#     spread = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'spread'].unstack()
#     spread = spread.loc[spread.index>='093000']
    
#     imbalance = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'imbalance'].unstack()
#     imbalance = imbalance.loc[imbalance.index>='093000']
    
#     h = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'high'].unstack()
#     l = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'low'].unstack()
#     h = h.loc[h.index>='093000']
#     l = l.loc[l.index>='093000']
#     hl = np.log(h) - np.log(l)
    
#     v = df_sql.set_index(['trade_time', 'stock_code']).loc[:, 'vol'].unstack()
#     v = v.loc[v.index>='093000']
    
#     # w_ = 
    
#     # df = DataFrame({'factor_value':s})
#     # df.loc[:, 'trade_date'] = trade_date
#     # df.loc[:, 'factor_name'] = 'momentum'
#     # df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#     # df.to_sql('tdailyhffactor', engine, schema='intermediate', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

# f_list = [Beta, Corrmarket, 
#           sigma, sigmaopen, skew, 
#           ETR, 
#           spread, imbalance, 
#           ]
# # f_list = [Corrmarket]
# #%%
# if __name__ == '__main__':
#     end_date = datetime.datetime.today().strftime('%Y%m%d')
#     start_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')
#     # start_date = '20210101'
#     # end_date = '20230829'
#     # start_date = '20230714'
#     engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    
#     sql_tmp = """
#     select trade_time, stock_code, 
#     open, high, low, close, spread, vol, amount, imbalance 
#     from mindata.tmindata{trade_date}
#     where trade_date = {trade_date}
#     """
    
#     trade_dates = tools.get_trade_cal(start_date, end_date)
    
#     for trade_date in trade_dates:
#         sql = sql_tmp.format(trade_date=trade_date)
#         df_sql = pd.read_sql(sql, engine)
#         for f in f_list:
#             print(trade_date, f)
#             f(df_sql)