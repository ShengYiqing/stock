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
end_date = '20230930'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r_d.unstack()
stock_codes = tuple(y.columns)
#%%
start_date_sql = tools.trade_date_shift(start_date, 60)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

sql = """
select t1.trade_date, t1.stock_code, 
t1.open, t1.close, t1.high, t1.low, t2.adj_factor, 
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
o = df.loc[:, 'open']
adj_factor = df.loc[:, 'adj_factor']
c = np.log(c * adj_factor).unstack()
o = np.log(o * adj_factor).unstack()

# r = c.diff()
r = c - o
# r = o - c.shift()
# h = df.loc[:, 'high']
# l = df.loc[:, 'low']
# u = df.loc[:, 'up_limit']
# d = df.loc[:, 'down_limit']

# ud = (u == h) | (d == l)
# ud = ud.unstack().fillna(False)
# r[ud] = np.nan

for i in range(4):
    x = r.rolling(5).sum().shift(5*i)
    x_ = DataFrame(x, index=y.index, columns=y.columns)
    x_[y.isna()] = np.nan
    tools.factor_analyse(x_, y, 10, 'x%s'%i)
