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
factor_dic = {
    'crhld': -1, 
    'crtrd': -1,
    
    'crhl2c': -1,
    'crhl2o': 1,
    
    'crsmvold': 1, 
    
    }
sql = tools.generate_sql_y_x(factor_dic.keys(), start_date, end_date, white_dic=None, is_trade=False, is_industry=False)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine)
df = df.set_index(['trade_date', 'stock_code']).loc[:, factor_dic.keys()]
df = df.groupby('trade_date').rank(pct=True)
for factor in factor_dic.keys():
    df.loc[:, factor] = df.loc[:, factor] * factor_dic[factor]
df = df.mean(1)
df = df.unstack()
df.index.name = 'trade_date'
df.columns.name = 'stock_code'
x_ = df
x_ = tools.neutralize(df, factors=['mc', 'bp'])
x_ = DataFrame(x_, index=y_a.index, columns=y_a.columns)
x_[y_a.isna()] = np.nan
tools.factor_analyse(x_, y_a, 7, 'cxx')
