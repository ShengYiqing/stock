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
start_date = '20180101'
end_date = '20230705'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r_d.unstack()
stock_codes = list(y.columns)
#%%
start_date_sql = tools.trade_date_shift(start_date, 250)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
factor_dic = {
    'crhl': -1, 
    'crtr': -1,
    
    
    }
sql = tools.generate_sql_y_x(factor_dic.keys(), start_date, end_date, white_dic=None, is_trade=False, is_industry=False)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine)
df = df.set_index(['trade_date', 'stock_code']).loc[:, factor_dic.keys()]
df = df.groupby('trade_date').rank(pct=True)
for factor in factor_dic.keys():
    df.loc[:, factor] = df.loc[:, factor] * factor_dic[factor]
df = df.mean(1)
df = df.unstack()
x_ = DataFrame(df, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 21, 'crx')
