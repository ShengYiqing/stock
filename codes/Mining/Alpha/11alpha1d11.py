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
end_date = '20231231'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r.unstack()
stock_codes = tuple(y.columns)
#%%
start_date_sql = tools.trade_date_shift(start_date, 1)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

sql = """
select t1.trade_date, t1.stock_code, 
t1.open, t1.close, t1.high, t1.low, t1.vol, t1.amount,
t2.adj_factor, 
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
o = np.log(df.loc[:, 'open'])
c = np.log(df.loc[:, 'close'])
h = np.log(df.loc[:, 'high'])
l = np.log(df.loc[:, 'low'])
x = (l-o).unstack() * (l-c).unstack()

# af = df.loc[:, 'adj_factor']
# r = np.log(c * af).unstack().diff()

# h = df.loc[:, 'high']
# l = df.loc[:, 'low']
# u = df.loc[:, 'up_limit']
# d = df.loc[:, 'down_limit']

# ud = (u == h) | (d == l)
# ud = ud.unstack().fillna(False)
# r[ud] = np.nan
# n1 = 5
# n2 = 250
# x1 = r.rolling(n1, min_periods=int(0.8*n1)).skew()
# x2 = r.rolling(n2, min_periods=int(0.8*n2)).skew()
# x = x1 - x2
# x.index.name = 'trade_date'
# x.columns.name = 'stock_code'
# x = tools.neutralize(x)
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 7, 'alpha1d11')
