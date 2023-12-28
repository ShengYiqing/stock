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
start_date = '20120101'
end_date = '20230830'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r_d.unstack()
stock_codes = tuple(y.columns)
#%%
start_date_sql = tools.trade_date_shift(start_date, 1250)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

sql = """
select t1.trade_date, t1.stock_code, 
t1.close, t1.high, t1.low, t2.adj_factor, 
tud.up_limit, tud.down_limit 
from ttsdaily t1
left join ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
left join ttsstklimit tud
on t1.stock_code = tud.stock_code
and t1.trade_date = tud.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and t1.stock_code in {stock_codes}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date, stock_codes=stock_codes)
df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
c = df.loc[:, 'close']
af = df.loc[:, 'adj_factor']
r = np.log(c * af).unstack().diff()

h = df.loc[:, 'high']
l = df.loc[:, 'low']
u = df.loc[:, 'up_limit']
d = df.loc[:, 'down_limit']

ud = (u == h) | (d == l)
ud = ud.unstack().fillna(False)
r[ud] = np.nan
dic = {}
n = 3
for i in range(1, 1+n):
    dic[i] = r.rolling(30, min_periods=int(0.8*30)).sum().shift(240*i-20).stack()
seasonality = DataFrame(dic).sum(1).unstack()
x = seasonality
# x.index.name = 'trade_date'
# x.columns.name = 'stock_code'
# x = tools.neutralize(x)
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 10, 'seasonality')
