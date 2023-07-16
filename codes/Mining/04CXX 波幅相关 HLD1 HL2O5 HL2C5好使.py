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

sql = """
select tl.trade_date, tl.stock_code, tl.r_d_a, tl.r_d_o, tl.r_d_c 
from label.tdailylabel tl
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
where tl.trade_date >= {start_date}
and tl.trade_date <= {end_date}
and tl.rank_mc > 0.9
and tl.rank_cmc > 0.382
and tl.rank_amount > 0.382
and tl.rank_price > 0.382
and tl.rank_revenue > 0.382
and ti.l3_name in {white_ind}
""".format(start_date=start_date, end_date=end_date, white_ind=tuple(gc.WHITE_INDUSTRY_LIST))
y_a = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_d_a.unstack()
y_o = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_d_o.unstack()
y_c = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_d_c.unstack()
y_c_s = y_c.shift(-1)
stock_codes = list(y_a.columns)
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
hl = (np.log(h) - np.log(l)).unstack()
hl2o = (np.log(h) + np.log(l) - 2 * np.log(o)).unstack()
hl2c = (np.log(h) + np.log(l) - 2 * np.log(c)).unstack()

x = r
w_dic = {'hl': hl, 'hl2o':hl2o, 'hl2c':hl2c}
n_list = [20, 60, 250]
for k in w_dic.keys():
    
    w = DataFrame(w_dic[k], index=r.index, columns=r.columns)
    
    # x = x.rank(axis=1, pct=True)
    w = w.diff()
    
    for n in n_list:
        ret = r.rolling(n, min_periods=int(0.618*n)).corr(w)
        
        x_ = DataFrame(ret, index=y_a.index, columns=y_a.columns)
        x_[y_a.isna()] = np.nan
        tools.factor_analyse(x_, y_a, 7, 'cr%s_%s_d'%(k, n))
