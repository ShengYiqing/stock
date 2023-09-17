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
n = 250
start_date_sql = tools.trade_date_shift(start_date, n)

sql = """
select t1.trade_date, t1.stock_code, 
1/t1.pb bp, 1/t1.pe_ttm ep, 1/t1.ps_ttm sp
from tsdata.ttsdailybasic t1
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and t1.stock_code in {stock_codes}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date, 
                 stock_codes=tuple(stock_codes))
df = pd.read_sql(sql, engine)

bp = df.set_index(['trade_date', 'stock_code']).loc[:, 'bp']
bp = np.log(bp).replace(-np.inf, np.nan)
ep = df.set_index(['trade_date', 'stock_code']).loc[:, 'ep']
ep = np.log(bp).replace(-np.inf, np.nan)
sp = df.set_index(['trade_date', 'stock_code']).loc[:, 'sp']
sp = np.log(sp).replace(-np.inf, np.nan)

# x = DataFrame({
#     'bp':bp, 
#     'ep':ep, 
#     'sp':sp, 
#     })

# ep = df.set_index(['trade_date', 'stock_code']).loc[:, 'ep']
# ep = np.log(ep).replace(-np.inf, np.nan)

# x = ep.unstack()
# x = tools.neutralize(x)
x = bp.unstack()
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 7, 'bp')
