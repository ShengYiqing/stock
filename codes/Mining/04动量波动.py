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
end_date = '20230830'
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
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
o = df.loc[:, 'open']
c = df.loc[:, 'close']
h = df.loc[:, 'high']
l = df.loc[:, 'low']
u = df.loc[:, 'up_limit']
d = df.loc[:, 'down_limit']

ud = (u == h) | (d == l)
ud = ud.unstack().fillna(False)
af = df.loc[:, 'adj_factor']

r = np.log(c * af).unstack().diff()
r[ud] = np.nan
oc = np.log(o * af).unstack() - np.log(c * af).unstack().shift()
co = (np.log(c) - np.log(o)).unstack()
hl = (np.log(h) - np.log(l)).unstack()
hl2o = (np.log(h) + np.log(l) - 2 * np.log(o)).unstack()
hl2c = (np.log(h) + np.log(l) - 2 * np.log(c)).unstack()

# r = r.rank(axis=1, pct=True)
w = hl #- hl.ewm(halflife=5).mean()

trade_dates = tools.get_trade_cal(start_date, end_date)

n = 250
n_q = 20
q_lists = [
    [i/n_q, (1+i)/n_q] for i in range(n_q)
    ]
q_lists = [
    [0.05, 1]
    ]
for q_list in q_lists:
    j = q_list[0]
    k = q_list[1]
    dic = {}
    for trade_date in trade_dates:
        print(trade_date, n, j, k)
        r_tmp = r.loc[r.index<=trade_date]
        w_tmp = w.loc[w.index<=trade_date]
        r_tmp = r_tmp.iloc[(-n):(-20)]
        w_tmp = w_tmp.iloc[(-n):(-20)]
        w_tmp = w_tmp.dropna(axis=1, thresh=0.618*n)
        w_tmp = w_tmp.rank(pct=True)
        w_tmp = ((w_tmp>=j)&(w_tmp<=k)).astype(int).replace(0, np.nan)
        dic[trade_date] = (r_tmp * w_tmp).mean().dropna()
    
    x = DataFrame(dic).T
    # x.index.name = 'trade_date'
    # x.columns.name = 'stock_code'
    # x = tools.neutralize(x)
    x_ = DataFrame(x, index=y.index, columns=y.columns)
    x_[y.isna()] = np.nan
    tools.factor_analyse(x_, y, 21, 'wrhls%s[%s,%s]'%(n, j, k))
