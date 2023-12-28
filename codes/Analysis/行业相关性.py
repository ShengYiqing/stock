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
start_date = '20210101'
end_date = '20231130'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x_ind([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'ind_name']).r.unstack()
ind_names = list(y.columns)
#%%
n = 250
start_date_sql = tools.trade_date_shift(start_date, n)

sql = """
select t1.trade_date, t1.stock_code, 
t1.close, t2.adj_factor
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
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and tl.days_list >= {days_list}
and tl.price >= {price}
and tl.amount >= {amount}
and ts.rank_mc >= {rank_mc}
""".format(start_date=start_date_sql, end_date=end_date, 
days_list=gc.LIMIT_DAYS_LIST, 
price=gc.LIMIT_PRICE, amount=gc.LIMIT_AMOUNT, 
rank_mc=gc.LIMIT_RANK_MC)

df = pd.read_sql(sql, engine)

c = df.set_index(['trade_date', 'stock_code']).loc[:, 'close']
adj_factor = df.set_index(['trade_date', 'stock_code']).loc[:, 'adj_factor']
r = np.log(c * adj_factor).groupby('stock_code').diff()
r = r.unstack()
r.index = pd.to_datetime(r.index)
r = r.resample('m').sum()
sql_ind = '''
select stock_code, l1_name, l2_name, l3_name 
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
    
l2_dic = {}
for i in range(len(df_stock_ind)):
    stock_code = df_stock_ind.iloc[i, 0]
    l2 = df_stock_ind.iloc[i, 2]
    
    if l2 not in l2_dic.keys():
        l2_dic[l2] = [stock_code]
    else:
        l2_dic[l2].append(stock_code)
      
l3_dic = {}
for i in range(len(df_stock_ind)):
    stock_code = df_stock_ind.iloc[i, 0]
    l3 = df_stock_ind.iloc[i, 3]
    
    if l3 not in l3_dic.keys():
        l3_dic[l3] = [stock_code]
    else:
        l3_dic[l3].append(stock_code)
    
r_l1 = DataFrame(
    {
     l1: r.loc[:, list(set(l1_dic[l1]).intersection(set(r.columns)))].mean(1)
     for l1 in l1_dic.keys()}
    )
   
r_l2 = DataFrame(
    {
     l1: r.loc[:, list(set(l2_dic[l1]).intersection(set(r.columns)))].mean(1)
     for l1 in l2_dic.keys()}
    )
   
r_l3 = DataFrame(
    {
     l1: r.loc[:, list(set(l3_dic[l1]).intersection(set(r.columns)))].mean(1)
     for l1 in l3_dic.keys()}
    )
r_ind = pd.concat([r_l1, r_l2, r_l3], axis=1)
r_ind.corr().to_csv('./corr.csv')
