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
n = 250
start_date_sql = tools.trade_date_shift(start_date, n+1)

sql = """
select trade_date, stock_code, 
turnover_rate from tsdata.ttsdailybasic
where trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine)

tr = df.set_index(['trade_date', 'stock_code']).loc[:, 'turnover_rate']

tr = np.log(tr).unstack()
trd = tr.diff()
tr_m = tr.ewm(halflife=5).mean()
tr_s = tr.ewm(halflife=5).std()

trd_s = trd.ewm(halflife=5).std()

x = tr_m

x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 7, 'corrhlr')
