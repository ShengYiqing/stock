import time
import datetime
import os
import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import Global_Config as gc
import tools
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sqlalchemy import create_engine


engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306")

end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
# start_date = '20100101'

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
and ti.l3_name in {ind_list}
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
ind_list=tuple(gc.WHITE_INDUSTRY_LIST),
days_list=gc.LIMIT_DAYS_LIST, 
price=gc.LIMIT_PRICE, amount=gc.LIMIT_AMOUNT, 
rank_mc=gc.LIMIT_RANK_MC, rank_pb=gc.LIMIT_RANK_PB, 
n_ind=gc.LIMIT_N_IND, n=gc.LIMIT_N)

df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
mc = df.style_mc.unstack()
r_a = df.r_a.unstack()
r_o = df.r_o.unstack()
r_c = df.r_c.unstack()

sql_ind = '''
select stock_code, l1_name, l3_name 
from indsw.tindsw
'''
df_stock_ind = pd.read_sql(sql_ind, engine).dropna()

l1_dic = {}
# l3_dic = {}
for i in range(len(df_stock_ind)):
    stock_code = df_stock_ind.iloc[i, 0]
    l1 = df_stock_ind.iloc[i, 1]
    # l3 = df_stock_ind.iloc[i, 2]
    
    if l1 not in l1_dic.keys():
        l1_dic[l1] = [stock_code]
    else:
        l1_dic[l1].append(stock_code)
    
    # if l3 not in l3_dic.keys():
    #     l3_dic[l3] = [stock_code]
    # else:
    #     l3_dic[l3].append(stock_code)

r_a_1 = DataFrame(
    {
     l1: r_a.loc[:, list(set(l1_dic[l1]).intersection(set(r_a.columns)))].mean(1)
     for l1 in l1_dic.keys()}
    )
r_o_1 = DataFrame(
    {
     l1: r_o.loc[:, list(set(l1_dic[l1]).intersection(set(r_o.columns)))].mean(1)
     for l1 in l1_dic.keys()}
    )
r_c_1 = DataFrame(
    {
     l1: r_c.loc[:, list(set(l1_dic[l1]).intersection(set(r_c.columns)))].mean(1)
     for l1 in l1_dic.keys()}
    )

# r_a_3 = DataFrame(
#     {
#      l3: r_a.loc[:, list(set(l3_dic[l3]).intersection(set(r_a.columns)))].mean(1)
#      for l3 in l3_dic.keys()}
#     )
# r_o_3 = DataFrame(
#     {
#      l3: r_o.loc[:, list(set(l3_dic[l3]).intersection(set(r_o.columns)))].mean(1)
#      for l3 in l3_dic.keys()}
#     )
# r_c_3 = DataFrame(
#     {
#      l3: r_c.loc[:, list(set(l3_dic[l3]).intersection(set(r_c.columns)))].mean(1)
#      for l3 in l3_dic.keys()}
#     )

df_1 = pd.concat({'r_a':r_a_1,  
                  'r_o':r_o_1, 
                  'r_c':r_c_1, 
                  }, axis=1)
df_1 = df_1.loc[df_1.index>=start_date]
df_1 = df_1.stack(dropna=False)
df_1.index.names = ['trade_date', 'ind_name']
# df_1.loc[:, 'level'] = 1
df_1.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
df_1.to_sql('tinddailylabel', engine, schema='label', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)

# df_3 = pd.concat({'r_a':r_a_3,  
#                   'r_o':r_o_3, 
#                   'r_c':r_c_3, 
#                   }, axis=1)
# df_3 = df_3.loc[df_3.index>=start_date]
# df_3 = df_3.stack(dropna=False)
# df_3.index.names = ['trade_date', 'ind_name']
# df_3.loc[:, 'level'] = 3
# df_3.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
# df_3.to_sql('tinddailylabel', engine, schema='label', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)
