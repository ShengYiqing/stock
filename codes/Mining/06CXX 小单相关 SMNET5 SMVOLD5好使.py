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

sql_y = tools.generate_sql_y_x([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r_d.unstack()
stock_codes = list(y.columns)
#%%
start_date_sql = tools.trade_date_shift(start_date, 250)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

sql = """
select t1.trade_date, t1.stock_code, 
t1.close, t2.adj_factor, 
tmf.buy_sm_vol, tmf.sell_sm_vol,  
t1.vol
from ttsdaily t1
left join ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
left join ttsmoneyflow tmf
on t1.stock_code = tmf.stock_code
and t1.trade_date = tmf.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
c = df.loc[:, 'close']
af = df.loc[:, 'adj_factor']
v = df.loc[:, 'vol']
buy_sm_vol = df.loc[:, 'buy_sm_vol']
sell_sm_vol = df.loc[:, 'sell_sm_vol']

sm_vol = ((buy_sm_vol + sell_sm_vol) / v).unstack()
sm_net = ((buy_sm_vol - sell_sm_vol) / v).unstack()

r = np.log(c * af).unstack().diff()

# hl = hl.rank(axis=1, pct=True)
#%%
# x = tr60
n_list = [5]
w_dic = {
    'sm_vol': sm_vol, 
    # 'sm_net': sm_net, 
    }
# w_list = [hl, ho, lo, ch, cl, hla, hloc, hl2o, hl2c]
for n in n_list:
    for w in w_dic.keys():
        x = r.ewm(halflife=n).corr(w_dic[w])
        x_ = DataFrame(x, index=y.index, columns=y.columns)
        x_[y.isna()] = np.nan
        tools.factor_analyse(x_, y, 21, 'cr%s_%s'%(w, n))
        
        x = r.ewm(halflife=n).corr(w_dic[w].shift())
        x_ = DataFrame(x, index=y.index, columns=y.columns)
        x_[y.isna()] = np.nan
        tools.factor_analyse(x_, y, 21, 'cr%s_%s_s'%(w, n))
        
        x = r.ewm(halflife=n).corr(w_dic[w].diff())
        x_ = DataFrame(x, index=y.index, columns=y.columns)
        x_[y.isna()] = np.nan
        tools.factor_analyse(x_, y, 21, 'cr%s_%s_d'%(w, n))
