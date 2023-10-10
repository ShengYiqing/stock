import datetime
import os
import sys
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import Global_Config as gc
import tools
from sqlalchemy import create_engine

#%%
start_date = '20180901'
end_date = '20230830'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r_d.unstack()
stock_codes = list(y.columns)
#%%
n = 250
start_date_sql = tools.trade_date_shift(start_date, n+1)
factor_dic = {'operation':1, 
              'gross':1, 
              'core':1, 
              'profitability':1, 
              'cash':1, 
              'growth':1, 
              'stability':1, 
              }
sql = tools.generate_sql_y_x(factor_dic.keys(), start_date, end_date, is_trade=False, is_industry=False)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine)
df = df.set_index(['trade_date', 'stock_code']).loc[:, factor_dic.keys()]
df = df.groupby('trade_date').rank(pct=True)
for factor in factor_dic.keys():
    # x = df.loc[:, factor].unstack()
    # x_ = DataFrame(x, index=y.index, columns=y.columns)
    # x_[y.isna()] = np.nan
    # tools.factor_analyse(x_, y, 21, factor)

    df.loc[:, factor] = df.loc[:, factor] * factor_dic[factor]
df = df.mean(1)
df = df.unstack()
df.index.name = 'trade_date'
df.columns.name = 'stock_code'
# df = tools.neutralize(df)

x = df

x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 10, 'quality')
