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


engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")


end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
# start_date = '20100101'

start_date_sql = tools.trade_date_shift(start_date, 250)

sql = """
select t1.stock_code, t1.trade_date, 
       t1.open, t1.high, t1.low, t1.close, t1.vol, t1.amount, 
       (t4.total_mv / t4.total_share * t4.free_share) mc, t4.pb pb, 
       t2.adj_factor, t3.suspend_type
from tsdata.ttsdaily t1
left join tsdata.ttsadjfactor t2
on t1.trade_date = t2.trade_date
and t1.stock_code = t2.stock_code
left join tsdata.ttssuspend t3
on t1.trade_date = t3.trade_date
and t1.stock_code = t3.stock_code
left join tsdata.ttsdailybasic t4
on t1.trade_date = t4.trade_date
and t1.stock_code = t4.stock_code
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and substr(t1.stock_code, 1, 1) not in ('4', '8')
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)

df = pd.read_sql(sql, engine)
df = df.set_index(['trade_date', 'stock_code']).unstack()
OPEN = df.loc[:, 'open']
CLOSE = df.loc[:, 'close']
HIGH = df.loc[:, 'high']
LOW = df.loc[:, 'low']
VOL = df.loc[:, 'vol']
AMOUNT = df.loc[:, 'amount']
mc = df.loc[:, 'mc']
ADJ = df.loc[:, 'adj_factor']
suspend = df.loc[:, 'suspend_type']

AVG = AMOUNT / VOL

AVG_hfq = np.log(AVG * ADJ).replace(-np.inf, np.nan)
open_hfq = np.log(OPEN * ADJ).replace(-np.inf, np.nan)
close_hfq = np.log(CLOSE * ADJ).replace(-np.inf, np.nan)

r_a = AVG_hfq.shift(-2) - AVG_hfq.shift(-1)

r_o = open_hfq.shift(-2) - open_hfq.shift(-1)

r_c = close_hfq.shift(-1) - close_hfq

yiziban = (HIGH == LOW).astype(int)
yiziban[HIGH.isna()] = np.nan
yiziban = yiziban.shift(-1)
yiziban.iloc[-1, :] = 0

suspend = suspend.copy()
suspend[suspend.notna()] = 1
suspend[suspend.isna()] = 0
suspend[CLOSE.isna()] = np.nan
suspend = suspend.shift(-1)
suspend.iloc[-1, :] = 0

days_new = 250
start_date_new = tools.trade_date_shift(start_date, days_new)
sql_new = """
select issue_date, stock_code, 1 as a from ttsnewshare
where issue_date >= {start_date_new}
""".format(start_date_new=start_date_new)
df_new = pd.read_sql(sql_new, engine)
df_new = df_new.set_index(['issue_date', 'stock_code']).a.unstack()
days_list = DataFrame(df_new, index=r_c.index).fillna(method='ffill').cumsum()
days_list.loc[:, list(set(r_c.columns) - set(days_list.columns))] = 65535


mc = mc
amount = AMOUNT.ewm(halflife=60, min_periods=60).mean()
price = CLOSE

df = pd.concat({'r_a':r_a,  
                'r_o':r_o, 
                'r_c':r_c, 
                'days_list':days_list,
                'suspend':suspend,
                'breaker':yiziban,
                'price':price,
                'amount':amount,
                'mc':mc,
                }, axis=1)
df = df.loc[df.index>=start_date]
df = df.stack()

df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/label?charset=utf8")
df.to_sql('tdailylabel', engine, schema='label', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)
