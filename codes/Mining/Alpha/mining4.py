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
import statsmodels.api as sm

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
and tl.is_white = 1
and ti.l3_name in {white_ind}
""".format(start_date=start_date, end_date=end_date, white_ind=tuple(gc.WHITE_INDUSTRY_LIST))
y = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_daily.unstack()
stock_codes = list(y.columns)
#%%
start_date_sql = tools.trade_date_shift(start_date, 60)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

sql = """
select t1.STOCK_CODE, t1.TRADE_DATE, t1.CLOSE, t1.AMOUNT, t2.ADJ_FACTOR from ttsdaily t1
left join ttsadjfactor t2
on t1.STOCK_CODE = t2.STOCK_CODE
and t1.TRADE_DATE = t2.TRADE_DATE
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine).set_index(['TRADE_DATE', 'STOCK_CODE'])
AMOUNT = df.loc[:, 'AMOUNT']
CLOSE = df.loc[:, 'CLOSE']
ADJ_FACTOR = df.loc[:, 'ADJ_FACTOR']
AMOUNT = AMOUNT.unstack()
CLOSE = CLOSE.unstack()
ADJ_FACTOR = ADJ_FACTOR.unstack()
CLOSE = CLOSE * ADJ_FACTOR
VOL = AMOUNT / CLOSE
VOL.replace(0, np.nan, inplace=True)
VOL = np.log(VOL)
CLOSE = np.log(CLOSE)
r = CLOSE.diff()
x = r.shift().ewm(halflife=20).corr(VOL, method='spearman')

#%%
# x = tr60
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 10, 'wm')
