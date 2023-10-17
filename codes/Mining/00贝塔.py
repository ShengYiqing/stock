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
stock_codes = list(y.columns)
#%%
n = 250
start_date_sql = tools.trade_date_shift(start_date, n)

sql = """
select t1.trade_date, t1.stock_code, 
t1.close, t2.adj_factor, 
(t3.total_mv / t3.total_share * t3.free_share) mc, 1 / t3.pb bp
from tsdata.ttsdaily t1
left join tsdata.ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
left join tsdata.ttsdailybasic t3
on t1.stock_code = t3.stock_code
and t1.trade_date = t3.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine)

c = df.set_index(['trade_date', 'stock_code']).loc[:, 'close']
mc = df.set_index(['trade_date', 'stock_code']).loc[:, 'mc'].groupby('trade_date').rank(pct=True).unstack()
bp = df.set_index(['trade_date', 'stock_code']).loc[:, 'bp'].groupby('trade_date').rank(pct=True).unstack()

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

for n in [5, 20, 60]:
    x = (r.ewm(halflife=n, min_periods=60).corr(r_m) * r.ewm(halflife=n, min_periods=60).std()).div(r_m.ewm(halflife=n, min_periods=60).std(), axis=0)
    
    x = tools.neutralize(x, ['mc', 'bp'])
    x_ = DataFrame(x, index=y.index, columns=y.columns)
    x_[y.isna()] = np.nan
    tools.factor_analyse(x_, y, 10, 'beta_%s'%n)
