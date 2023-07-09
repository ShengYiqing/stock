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
from sqlalchemy import create_engine


engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")


end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
# start_date = '20100101'

start_date_sql = tools.trade_date_shift(start_date, 60)

sql = """
select t1.stock_code, t1.trade_date, 
       t1.open, t1.high, t1.low, t1.close, t1.vol, t1.amount, 
       t4.total_mv mc, t4.circ_mv circ_mc, 
       (t4.total_mv / t4.ps_ttm) s, 
       t2.adj_factor, t3.suspend_type
from ttsdaily t1
left join ttsadjfactor t2
on t1.trade_date = t2.trade_date
and t1.stock_code = t2.stock_code
left join ttssuspend t3
on t1.trade_date = t3.trade_date
and t1.stock_code = t3.stock_code
left join ttsdailybasic t4
on t1.trade_date = t4.trade_date
and t1.stock_code = t4.stock_code
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and substr(t1.stock_code, 1, 1) not in ('4', '8')
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)

df = pd.read_sql(sql, engine)
df = df.set_index(['trade_date', 'stock_code']).unstack()
CLOSE = df.loc[:, 'close']
HIGH = df.loc[:, 'high']
LOW = df.loc[:, 'low']
VOL = df.loc[:, 'vol']
AMOUNT = df.loc[:, 'amount']
mc = df.loc[:, 'mc']
circ_mc = df.loc[:, 'circ_mc']
s = df.loc[:, 's']
ADJ = df.loc[:, 'adj_factor']
suspend = df.loc[:, 'suspend_type']

AVG = AMOUNT / VOL

AVG_hfq = np.log(AVG * ADJ)

r_d = AVG_hfq.shift(-2) - AVG_hfq.shift(-1)
r_w = AVG_hfq.shift(-6) - AVG_hfq.shift(-1)
r_m = AVG_hfq.shift(-21) - AVG_hfq.shift(-1)

yiziban = (HIGH == LOW).astype(int)
yiziban = yiziban.shift(-1)
yiziban.iloc[-1, :] = 0

suspend = suspend.copy()
suspend[suspend.notna()] = 1
suspend.fillna(0, inplace=True)
suspend = suspend.shift(-1)
suspend.iloc[-1, :] = 0

days_new = 20
start_date_new = tools.trade_date_shift(start_date, days_new)
trade_dates_new = tools.get_trade_cal(start_date_new, end_date)
sql_new = """
select issue_date, stock_code, 1 as a from ttsnewshare
where issue_date >= {start_date_new}
""".format(start_date_new=start_date_new)
df_new = pd.read_sql(sql_new, engine)
df_new = df_new.set_index(['issue_date', 'stock_code']).unstack()
new = DataFrame(df_new.loc[:, 'a'], index=trade_dates_new)
new.fillna(method='bfill', inplace=True)
new.fillna(method='ffill', limit=days_new, inplace=True)

new = DataFrame(new, index=r_d.index, columns=r_d.columns)
new.fillna(0, inplace=True)

is_trade = yiziban + suspend + new

is_trade[CLOSE.isna()] = 1
is_trade[is_trade>0] = 1

is_trade = 1 - is_trade
rank_amount = AMOUNT.rolling(250, min_periods=20).mean().rank(axis=1, pct=True)
rank_price = CLOSE.rank(axis=1, pct=True)
rank_revenue = s.rank(axis=1, pct=True)
rank_cmc = circ_mc.rank(axis=1, pct=True)
rank_mc = mc.rank(axis=1, pct=True)
df = pd.concat({'R_DAILY':r_d, 
                'R_WEEKLY':r_w, 
                'R_MONTHLY':r_m, 
                'IS_TRADE':is_trade,
                'rank_amount':rank_amount,
                'rank_price':rank_price,
                'rank_revenue':rank_revenue,
                'rank_cmc':rank_cmc,
                'rank_mc':rank_mc,
                }, axis=1)
df = df.loc[df.index>=start_date]
df = df.stack()

df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/label?charset=utf8")
df.to_sql('tdailylabel', engine, schema='label', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)
