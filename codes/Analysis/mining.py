import datetime
import os
import sys
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
from sqlalchemy import create_engine

#%%
start_date = '20120101'
end_date = '20230505'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql = """
select tl.trade_date, tl.stock_code, tl.r_daily
from label.tdailylabel tl
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
where tl.trade_date >= {start_date}
and tl.trade_date <= {end_date}
and tl.is_trade = 1
and ti.l3_name in {white_ind}
""".format(start_date=start_date, end_date=end_date, white_ind=tuple(gc.WHITE_INDUSTRY_LIST))
y = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_daily.unstack()
stock_codes = list(y.columns)
#%%
n = 60
start_date_sql = tools.trade_date_shift(start_date, n+1)

sql = """
select t1.trade_date, t1.stock_code, 
t1.open, t1.high, t1.low, t1.close, tdb.turnover_rate tr, 
t1.vol, t1.amount, 
t2.adj_factor from tsdata.ttsdaily t1
left join tsdata.ttsdailybasic tdb
on t1.stock_code = tdb.stock_code
and t1.trade_date = tdb.trade_date
left join tsdata.ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine)

# tr = df.set_index(['trade_date', 'stock_code']).loc[:, 'tr']
# h = df.set_index(['trade_date', 'stock_code']).loc[:, 'high']
# l = df.set_index(['trade_date', 'stock_code']).loc[:, 'low']
v = df.set_index(['trade_date', 'stock_code']).loc[:, 'vol']
a = df.set_index(['trade_date', 'stock_code']).loc[:, 'amount']

c = df.set_index(['trade_date', 'stock_code']).loc[:, 'close']

# hl = np.log(h) - np.log(l)

adj_factor = df.set_index(['trade_date', 'stock_code']).loc[:, 'adj_factor']
p = a / v * adj_factor
r = np.log(c * adj_factor).groupby('stock_code').diff()

# w = hl.unstack()
w = (r**2).unstack()
r = r.unstack()
r = r.sub(r.mean(1), 0)
trade_dates = tools.get_trade_cal(start_date, end_date)
dic = {}
for trade_date in trade_dates:
    print(trade_date)
    r_tmp = r.loc[r.index<=trade_date]
    w_tmp = w.loc[w.index<=trade_date]
    r_tmp = r_tmp.iloc[-n:]
    w_tmp = w_tmp.iloc[-n:]
    w_tmp = w_tmp.dropna(axis=1)
    w_tmp = 0.5 - w_tmp.rank(pct=True)
    
    dic[trade_date] = (r_tmp * w_tmp).mean().dropna()

x = DataFrame(dic).T

#%%

x_1 = DataFrame(x, index=y.index, columns=y.columns)
x_1[y.isna()] = np.nan
tools.factor_analyse(x_1, y, 10, 'wm')
x_2 = tools.neutralize(x)
x_2 = DataFrame(x_2, index=y.index, columns=y.columns)
x_2[y.isna()] = np.nan
tools.factor_analyse(x_2, y, 10, 'wm')