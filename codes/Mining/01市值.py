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
end_date = '20231130'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x([], start_date, end_date, white_dic=None, style_dic=None, n_ind=500, n=5000)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r.unstack()
stock_codes = list(y.columns)
#%%
n = 250
start_date_sql = tools.trade_date_shift(start_date, n)

sql = """
select t1.trade_date, t1.stock_code, 
t1.total_mv mc 
from tsdata.ttsdailybasic t1
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and t1.stock_code in {stock_codes}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date, 
                 stock_codes=tuple(stock_codes))
df = pd.read_sql(sql, engine)

mc = df.set_index(['trade_date', 'stock_code']).loc[:, 'mc']
mc = np.log(mc)

x = mc.unstack()
# x = tools.neutralize(x)
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 10, 'ya')
