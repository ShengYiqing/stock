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
start_date = '20160101'
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

hl = (np.log(h) - np.log(l)).unstack()
hl2o = (np.log(h) + np.log(l) - 2 * np.log(o)).unstack()
hl2c = (np.log(h) + np.log(l) - 2 * np.log(c)).unstack()



# hl = hl.rank(axis=1, pct=True)
#%%
# x = tr60
n_list = [1, 5]
w_dic = {
    'hl': hl, 
    'hl2o': hl2o, 
    'hl2c': hl2c, 
    }
# w_list = [hl, ho, lo, ch, cl, hla, hloc, hl2o, hl2c]
for n in n_list:
    for w in w_dic.keys():
        x = r.ewm(halflife=n).corr(w_dic[w])
        x_ = DataFrame(x, index=y.index, columns=y.columns)
        x_[y.isna()] = np.nan
        tools.factor_analyse(x_, y, 7, 'cr%s_%s'%(w, n))

for n in n_list:
    for w in w_dic.keys():
        x = r.ewm(halflife=n).corr(w_dic[w].shift())
        x_ = DataFrame(x, index=y.index, columns=y.columns)
        x_[y.isna()] = np.nan
        tools.factor_analyse(x_, y, 7, 'cr%s_%s_s'%(w, n))

for n in n_list:
    for w in w_dic.keys():
        x = r.ewm(halflife=n).corr(w_dic[w].diff())
        x_ = DataFrame(x, index=y.index, columns=y.columns)
        x_[y.isna()] = np.nan
        tools.factor_analyse(x_, y, 7, 'cr%s_%s_d'%(w, n))
