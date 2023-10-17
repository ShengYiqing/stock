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

#%%
start_date_sql = tools.trade_date_shift(start_date, 250)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306")

sql = """
select t1.trade_date, t1.stock_code, 
t1.high, t1.low, t1.close, 
t2.adj_factor , t3.turnover_rate, 
tud.up_limit, tud.down_limit 
from tsdata.ttsdaily t1
left join tsdata.ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
left join tsdata.ttsdailybasic t3
on t1.stock_code = t3.stock_code
and t1.trade_date = t3.trade_date
left join tsdata.ttsstklimit tud
on t1.stock_code = tud.stock_code
and t1.trade_date = tud.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
c = df.loc[:, 'close']
h = df.loc[:, 'high']
l = df.loc[:, 'low']
af = df.loc[:, 'adj_factor']
r = np.log(c * af).unstack().diff()

h = df.loc[:, 'high']
l = df.loc[:, 'low']
u = df.loc[:, 'up_limit']
d = df.loc[:, 'down_limit']

ud = (u == h) | (d == l)
ud = ud.unstack().fillna(False)
r[ud] = np.nan

tr = df.loc[:, 'turnover_rate'].unstack()

hl = (np.log(h) - np.log(l)).unstack()

w1 = hl
w2 = tr

w_dic = {
    'w1': w1, 
    # 'w2': w2, 
    }

q_dic = {
    'w1': [0, 0.8], 
    # 'w2': [0, 0.8], 
    }

trade_dates = tools.get_trade_cal(start_date, end_date)

n = 220
dic = {}
for trade_date in trade_dates:
    print(trade_date, n, q_dic)
    r_tmp = r.loc[r.index<=trade_date].iloc[(-n):(-20)].dropna(axis=1, thresh=0.618*n)
    w_dic_tmp = {}
    for w_k in w_dic.keys():
        w_tmp = w_dic[w_k].loc[r_tmp.index, r_tmp.columns].rank(pct=True)
        w_dic_tmp[w_k] = ((w_tmp>=q_dic[w_k][0])&(w_tmp<=q_dic[w_k][1])).astype(int).stack()
    w_tmp = DataFrame(w_dic_tmp).mean(1).unstack()
    dic[trade_date] = (r_tmp * w_tmp).mean().dropna()

x = DataFrame(dic).T
# x.index.name = 'trade_date'
# x.columns.name = 'stock_code'
# x = tools.neutralize(x)
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 10, 'momentum')
