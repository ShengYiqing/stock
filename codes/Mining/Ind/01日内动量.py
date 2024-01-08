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
end_date = '20231231'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x_ind([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'ind_name']).r.unstack()
ind_names = list(y.columns)
#%%
n = 250
start_date_sql = tools.trade_date_shift(start_date, n)

sql = """
select t.* from
(
select t.*, rank() over(partition by trade_date order by mc desc) lead_stock from
(
select t.*, rank() over(partition by trade_date, l1_name, l2_name, l3_name order by mc desc) lead_ind from
(
select t1.trade_date, t1.stock_code, 
t1.open, t1.close, t2.adj_factor, 
ts.mc, ti.l1_name, ti.l2_name, ti.l3_name
from tsdata.ttsdaily t1
left join tsdata.ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
left join label.tdailylabel tl
on t1.stock_code = tl.stock_code
and t1.trade_date = tl.trade_date
left join style.tdailystyle ts
on t1.stock_code = ts.stock_code
and t1.trade_date = ts.trade_date
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and tl.days_list >= {days_list}
and tl.price >= {price}
and tl.amount >= {amount}
and ts.rank_mc >= {rank_mc}
and ts.rank_pb >= {rank_pb}
) t
) t
where lead_ind <= {n_ind}
) t
where lead_stock <= {n}
""".format(start_date=start_date_sql, end_date=end_date, 
days_list=gc.LIMIT_DAYS_LIST, 
price=gc.LIMIT_PRICE, amount=gc.LIMIT_AMOUNT, 
rank_mc=gc.LIMIT_RANK_MC, rank_pb=gc.LIMIT_RANK_PB, 
n_ind=gc.LIMIT_N_IND, n=gc.LIMIT_N)

df = pd.read_sql(sql, engine)

o = df.set_index(['trade_date', 'stock_code']).loc[:, 'open']
c = df.set_index(['trade_date', 'stock_code']).loc[:, 'close']
r = np.log(c / o)
r = r.unstack()

sql_ind = '''
select stock_code, l1_name, l3_name 
from indsw.tindsw
'''
df_stock_ind = pd.read_sql(sql_ind, engine).dropna()

l1_dic = {}
for i in range(len(df_stock_ind)):
    stock_code = df_stock_ind.iloc[i, 0]
    l1 = df_stock_ind.iloc[i, 1]
    
    if l1 not in l1_dic.keys():
        l1_dic[l1] = [stock_code]
    else:
        l1_dic[l1].append(stock_code)
    
r = DataFrame(
    {
     l1: r.loc[:, list(set(l1_dic[l1]).intersection(set(r.columns)))].mean(1)
     for l1 in l1_dic.keys()}
    )

n = 20
x = r.rolling(n, min_periods=int(0.8*n)).sum()

x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 5, 'intraday_%s'%n)

    
