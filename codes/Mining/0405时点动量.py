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
start_date = '20200101'
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
n = 250
start_date_sql = tools.trade_date_shift(start_date, n+1)

sql = """
select t1.trade_date, t1.stock_code, 
t1.open, t1.high, t1.low, t1.close, tdb.turnover_rate tr, 
t1.vol, t1.amount, 
t2.adj_factor from tsdata.ttsdaily t1
left join tsdata.ttsdailybasic tdb
on t1.stock_code = tdb.stock_code
and t1.trade_date = tdb.trade_date
left join tsdata.ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine)

# tr = df.set_index(['trade_date', 'stock_code']).loc[:, 'tr']
# h = df.set_index(['trade_date', 'stock_code']).loc[:, 'high']
# l = df.set_index(['trade_date', 'stock_code']).loc[:, 'low']
# v = df.set_index(['trade_date', 'stock_code']).loc[:, 'vol']
# a = df.set_index(['trade_date', 'stock_code']).loc[:, 'amount']

c = df.set_index(['trade_date', 'stock_code']).loc[:, 'close']

# hl = np.log(h) - np.log(l)

adj_factor = df.set_index(['trade_date', 'stock_code']).loc[:, 'adj_factor']
# p = a / v * adj_factor
r = np.log(c * adj_factor).groupby('stock_code').diff().unstack()
r = r.sub(r.mean(1), axis=0)

#%%
n = 250
start_date_sql = tools.trade_date_shift(start_date, n+1)

sql_i = """
select trade_date, 
open, high, low, close, vol, amount
from tsdata.ttsindexdaily
where trade_date >= {start_date}
and trade_date <= {end_date}
and index_name = '沪深300'
""".format(start_date=start_date_sql, end_date=end_date)
df_i = pd.read_sql(sql_i, engine).set_index('trade_date')

o_i = np.log(df_i.open)
h_i = np.log(df_i.high)
l_i = np.log(df_i.low)
c_i = np.log(df_i.close)
v_i = np.log(df_i.vol)
a_i = np.log(df_i.amount)

hl_i = h_i - l_i
oc_i = o_i - c_i.shift()
r_i = c_i.diff()

cp_r_i = np.abs(r_i - r_i.ewm(halflife=20).mean()) / r_i.ewm(halflife=20).std()
cp_c_i = np.abs(c_i - c_i.ewm(halflife=20).mean()) / c_i.ewm(halflife=20).std()

# n_a = 20
# a_i = r_i - r_i.ewm(halflife=n_a).mean()

d = r.std(1).rolling(250, min_periods=60).rank(pct=True)
n_list = [20]
w_dic = {
    # 'cp_r_i': cp_r_i, 
    # 'cp_c_i': cp_c_i, 
    'd': d, 
    }
# w_list = [hl, ho, lo, ch, cl, hla, hloc, hl2o, hl2c]
for n in n_list:
    for w in w_dic.keys():
        x = r.ewm(halflife=n).corr(w_dic[w])
        x = x * r.ewm(halflife=n).std()
        # x = x.div(w_dic[w].ewm(halflife=n).std(), axis=0)
        x = x.replace(-np.inf, np.nan).replace(np.inf, np.nan)
        x_ = DataFrame(x, index=y.index, columns=y.columns)
        x_[y.isna()] = np.nan
        tools.factor_analyse(x_, y, 3, 'cr%s_%s'%(w, n))
        
        x = r.ewm(halflife=n).corr(w_dic[w].shift())
        x = x * r.ewm(halflife=n).std()
        # x = x / w_dic[w].shift().ewm(halflife=n).std()
        x = x.replace(-np.inf, np.nan).replace(np.inf, np.nan)
        x_ = DataFrame(x, index=y.index, columns=y.columns)
        x_[y.isna()] = np.nan
        tools.factor_analyse(x_, y, 3, 'cr%s_%s_s'%(w, n))
        
        x = r.ewm(halflife=n).corr(w_dic[w].diff())
        x = x * r.ewm(halflife=n).std()
        # x = x / w_dic[w].diff().ewm(halflife=n).std()
        x = x.replace(-np.inf, np.nan).replace(np.inf, np.nan)
        x_ = DataFrame(x, index=y.index, columns=y.columns)
        x_[y.isna()] = np.nan
        tools.factor_analyse(x_, y, 3, 'cr%s_%s_d'%(w, n))
