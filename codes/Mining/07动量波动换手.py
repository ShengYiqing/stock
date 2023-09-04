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
start_date = '20220101'
end_date = '20230820'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r_d.unstack()

#%%
start_date_sql = tools.trade_date_shift(start_date, 250)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306")

sql = """
select t1.trade_date, t1.stock_code, 
t1.open, t1.high, t1.low, t1.close, t1.vol, t1.amount,
t2.adj_factor , t3.turnover_rate, ((t4.buy_sm_vol + t4.sell_sm_vol) / t1.vol) sm_vol_ratio, 
t5.factor_value beta
from tsdata.ttsdaily t1
left join tsdata.ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
left join tsdata.ttsdailybasic t3
on t1.stock_code = t3.stock_code
and t1.trade_date = t3.trade_date
left join tsdata.ttsmoneyflow t4
on t1.stock_code = t4.stock_code
and t1.trade_date = t4.trade_date
left join intermediate.tdailyhffactor t5
on t1.stock_code = t5.stock_code
and t1.trade_date = t5.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and t5.factor_name = 'corrmarket'
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
o = df.loc[:, 'open']
c = df.loc[:, 'close']
h = df.loc[:, 'high']
l = df.loc[:, 'low']
v = df.loc[:, 'vol']
a = df.loc[:, 'amount']
af = df.loc[:, 'adj_factor']
sm = np.log(df.loc[:, 'sm_vol_ratio']).unstack()
beta = np.log(df.loc[:, 'beta']).unstack()
avg = a / v
r = np.log(c * af).unstack().diff()
tr = df.loc[:, 'turnover_rate']
tr = np.log(tr).replace(-np.inf, np.nan).unstack()

hl = (np.log(h) - np.log(l)).unstack()

tr = DataFrame(tr, index=r.index, columns=r.columns)

w1 = hl - hl.rolling(5, min_periods=1).mean()
w2 = tr - tr.rolling(5, min_periods=1).mean()
w3 = sm - sm.rolling(5, min_periods=1).mean()
w4 = beta
w_dic = {
    'w1': w1, 
    'w2': w2, 
    'w3': w3, 
    }
q_dic = {
    'w1': [0.05, 0.8], 
    'w2': [0.05, 0.8], 
    'w3': [0.2, 0.95], 
    
    }


w_dic = {
    'w4': w4,  
    }
q_dic = {
    'w4': [0.2, 1], 
    
    }
trade_dates = tools.get_trade_cal(start_date, end_date)

n = 250
dic = {}
for trade_date in trade_dates:
    print(trade_date, n, q_dic)
    r_tmp = r.loc[r.index<=trade_date]
    r_tmp = r_tmp.iloc[(-n):(-20)]
    w_dic_tmp = {}
    for w_k in w_dic.keys():
        w_v = w_dic[w_k].loc[w_dic[w_k].index<=trade_date].iloc[(-n):(-20)].dropna(axis=1, thresh=0.618*n).rank(pct=True)
        w_dic_tmp[w_k] = ((w_v>=q_dic[w_k][0])&(w_v<=q_dic[w_k][1])).astype(int).stack()
    w_tmp = DataFrame(w_dic_tmp).sum(1).unstack()
    dic[trade_date] = (r_tmp * w_tmp).mean().dropna()

x = DataFrame(dic).T
# x.index.name = 'trade_date'
# x.columns.name = 'stock_code'
# x = tools.neutralize(x)
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 21, 'momentum')
