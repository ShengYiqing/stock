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

sql = """
select t1.trade_date, t1.stock_code, 
t1.open, t1.high, t1.low, t1.close, 
t2.adj_factor 
from ttsdaily t1
left join ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
o = df.loc[:, 'open']
c = df.loc[:, 'close']
h = df.loc[:, 'high']
l = df.loc[:, 'low']
af = df.loc[:, 'adj_factor']
r = np.log(c * af).unstack().diff()

oc = np.log(o * af).unstack() - np.log(c * af).unstack().shift()
co = (np.log(c) - np.log(o))
hl = (np.log(h) - np.log(l))
hl2o = (np.log(h) + np.log(l) - 2 * np.log(o))
hl2c = (np.log(h) + np.log(l) - 2 * np.log(c))

w = oc.shift()

# w = DataFrame({
#     'hl':-hl, 
#     'hl2o':hl2o, 
#     'hl2c':-hl2c,
#     }).groupby('trade_date').rank(pct=True).mean(1).unstack()

x = r.ewm(halflife=5).corr(w)

x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 7, 'crhl')
