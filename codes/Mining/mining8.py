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
import statsmodels.api as sm

#%%
start_date = '20120101'
end_date = '20230505'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql = """
select tl.trade_date, tl.stock_code, tl.r_daily
from label.tdailylabel tl
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
where tl.trade_date >= {start_date}
and tl.trade_date <= {end_date}
and tl.is_white = 1
and ti.l3_name in {white_ind}
""".format(start_date=start_date, end_date=end_date, white_ind=tuple(gc.WHITE_INDUSTRY_LIST))
y = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_daily.unstack()
stock_codes = list(y.columns)
#%%
start_date_sql = tools.trade_date_shift(start_date, 250)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

sql = """
select t1.trade_date, t1.stock_code, 
t1.close, tmf.buy_sm_vol, tmf.sell_sm_vol,  
t1.vol, 
t2.adj_factor from tsdata.ttsdaily t1
left join tsdata.ttsmoneyflow tmf
on t1.stock_code = tmf.stock_code
and t1.trade_date = tmf.trade_date
left join tsdata.ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine)

v = df.set_index(['trade_date', 'stock_code']).loc[:, 'vol']
buy_sm_vol = df.set_index(['trade_date', 'stock_code']).loc[:, 'buy_sm_vol']
sell_sm_vol = df.set_index(['trade_date', 'stock_code']).loc[:, 'sell_sm_vol']

sm_vol_rate = (buy_sm_vol + sell_sm_vol) / v

c = df.set_index(['trade_date', 'stock_code']).loc[:, 'close']

adj_factor = df.set_index(['trade_date', 'stock_code']).loc[:, 'adj_factor']

r = np.log(c * adj_factor).groupby('stock_code').diff()
r = r.unstack()
s = np.log(sm_vol_rate.replace(-np.inf, np.nan).replace(np.inf, np.nan).replace(0, np.nan)).unstack()
#%%
# x = tr60
x = r.ewm(halflife=5).corr(s)
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 10, 'crs')
