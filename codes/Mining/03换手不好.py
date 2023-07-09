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
start_date = '20200101'
end_date = '20230605'
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

x = trd_s

x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 7, 'corrhlr')
