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
       t4.total_mv mc, t4.pb pb, 
       t2.adj_factor, t3.suspend_type, 
       t5.l3_name
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
left join indsw.tindsw t5
on t1.stock_code = t5.stock_code
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
pb = df.loc[:, 'pb']
ADJ = df.loc[:, 'adj_factor']
suspend = df.loc[:, 'suspend_type']

AVG = AMOUNT / VOL

AVG_hfq = np.log(AVG * ADJ)
open_hfq = np.log(OPEN * ADJ)
close_hfq = np.log(CLOSE * ADJ)

r_d_a = AVG_hfq.shift(-2) - AVG_hfq.shift(-1)
r_w_a = AVG_hfq.shift(-6) - AVG_hfq.shift(-1)
r_m_a = AVG_hfq.shift(-21) - AVG_hfq.shift(-1)

r_d_o = open_hfq.shift(-2) - open_hfq.shift(-1)
r_w_o = open_hfq.shift(-6) - open_hfq.shift(-1)
r_m_o = open_hfq.shift(-21) - open_hfq.shift(-1)

r_d_c = close_hfq.shift(-1) - close_hfq
r_w_c = close_hfq.shift(-5) - close_hfq
r_m_c = close_hfq.shift(-20) - close_hfq

yiziban = (HIGH == LOW).astype(int)
yiziban = yiziban.shift(-1)
yiziban.iloc[-1, :] = 0

suspend = suspend.copy()
suspend[suspend.notna()] = 1
suspend.fillna(0, inplace=True)
suspend = suspend.shift(-1)
suspend.iloc[-1, :] = 0

days_new = 60
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

new = DataFrame(new, index=r_d_a.index, columns=r_d_a.columns)
new.fillna(0, inplace=True)

is_trade = yiziban + suspend + new

is_trade[CLOSE.isna()] = 1
is_trade[is_trade>0] = 1

is_trade = 1 - is_trade

sql = """
select trade_date, close from ttsindexdaily
where trade_date >= {start_date}
and trade_date <= {end_date}
and index_name = '中证800'
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
close_m = pd.read_sql(sql, engine).set_index('trade_date').loc[:, 'close']
r_m = np.log(close_m).diff()

r = close_hfq.diff()
beta = r.ewm(halflife=60, min_periods=60).corr(r_m) * r.ewm(halflife=60, min_periods=60).std()

beta.replace(np.inf, np.nan, inplace=True)
beta.replace(-np.inf, np.nan, inplace=True)

rank_beta = beta.rank(axis=1, pct=True)
rank_mc = mc.rank(axis=1, pct=True)
rank_pb = pb.rank(axis=1, pct=True)
rank_amount = AMOUNT.ewm(halflife=60, min_periods=60).mean().rank(axis=1, pct=True)
rank_price = CLOSE.rank(axis=1, pct=True)

# logmc = np.log(mc).ewm(halflife=60, min_periods=60).mean()
# logbp = np.log(bp).ewm(halflife=60, min_periods=60).mean()
# l3 = df.loc[:, 'l3_name']

# def tune(data):
#     X = pd.concat([pd.get_dummies(data.loc[:, 'l3']), data.loc[:, ['mc', 'bp']]], axis=1)
#     X.loc[:, 'mc'] = tools.winsorize(X.loc[:, 'mc'])
#     X.loc[:, 'bp'] = tools.winsorize(X.loc[:, 'bp'])
#     X = sm.add_constant(X)
    
#     y = tools.winsorize(data.loc[:, 'y'])
    
#     y_predict = X.dot(np.linalg.inv(X.T.dot(X)+0.001*np.identity(len(X.T))).dot(X.T).dot(y))
    
#     return y_predict

# data = DataFrame({'y': beta.stack(), 
#                   'mc': logmc.stack(), 
#                   'bp': logbp.stack(), 
#                   'l3': l3.stack(), 
#                   }).dropna()
# beta_tune = data.groupby('trade_date', as_index=False).apply(tune).reset_index(0, drop=True).unstack()


# rank_beta = beta.rank(axis=1, pct=True)
# rank_beta_tune = beta_tune.rank(axis=1, pct=True)

df = pd.concat({'r_d_a':r_d_a, 
                'r_w_a':r_w_a, 
                'r_m_a':r_m_a, 
                'r_d_o':r_d_o, 
                'r_w_o':r_w_o, 
                'r_m_o':r_m_o, 
                'r_d_c':r_d_c, 
                'r_w_c':r_w_c, 
                'r_m_c':r_m_c, 
                'is_trade':is_trade,
                'rank_price':rank_price,
                'rank_amount':rank_amount,
                'rank_beta':rank_beta,
                'rank_mc':rank_mc,
                'rank_pb':rank_pb,
                }, axis=1)
df = df.loc[df.index>=start_date]
df = df.stack()

df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/label?charset=utf8")
df.to_sql('tdailylabel', engine, schema='label', index=True, if_exists='append', chunksize=5000, method=tools.mysql_replace_into)
