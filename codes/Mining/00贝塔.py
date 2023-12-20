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
end_date = '20231130'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r.unstack()
stock_codes = list(y.columns)
#%%
n = 20
start_date_sql = tools.trade_date_shift(start_date, n)

sql = """
select t1.trade_date, t1.stock_code, 
t1.close, t2.adj_factor
from tsdata.ttsdaily t1
left join tsdata.ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and t1.stock_code in {stock_codes}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date, 
                 stock_codes=tuple(stock_codes))
df = pd.read_sql(sql, engine)

c = df.set_index(['trade_date', 'stock_code']).loc[:, 'close']
adj_factor = df.set_index(['trade_date', 'stock_code']).loc[:, 'adj_factor']
r = np.log(c * adj_factor).groupby('stock_code').diff()
r = r.unstack()

sql = """
select trade_date, close from tsdata.ttsindexdaily
where trade_date >= {start_date}
and trade_date <= {end_date}
and index_name = '沪深300'
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
close_m = pd.read_sql(sql, engine).set_index('trade_date').loc[:, 'close']
r_m = np.log(close_m).diff()

n = 20
sxy = r.rolling(n, min_periods=int(0.618*n)).cov(r_m)
sxx = r_m.rolling(n, min_periods=int(0.618*n)).var()
# sxy = r.ewm(halflife=n, min_periods=int(0.618*n)).cov(r_m)
# sxx = r_m.ewm(halflife=n, min_periods=int(0.618*n)).var()
cxy = r.rolling(n, min_periods=int(0.618*n)).corr(r_m)
x = sxy.div(sxx, axis=0).replace(-np.inf, np.nan).replace(np.inf, np.nan)
x = tools.neutralize(x, ind='l3')
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 10, 'beta_%s'%n)
