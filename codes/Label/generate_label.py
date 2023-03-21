import time
import datetime
import os
import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
sys.path.append('../../Codes')
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)

import Global_Config as gc
import tools
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")


end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
# start_date = '20100101'


sql = """
select t1.stock_code, t1.trade_date, 
       t1.open, t1.high, t1.low, t1.close, t1.vol, t1.amount, 
       t2.adj_factor, t3.suspend_type
from ttsdaily t1
left join ttsadjfactor t2
on t1.trade_date = t2.trade_date
and t1.stock_code = t2.stock_code
left join ttssuspend t3
on t1.trade_date = t3.trade_date
and t1.stock_code = t3.stock_code
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date, end_date=end_date)

df = pd.read_sql(sql, engine)
df = df.set_index(['trade_date', 'stock_code']).unstack()
CLOSE = df.loc[:, 'close']
HIGH = df.loc[:, 'high']
LOW = df.loc[:, 'low']
VOL = df.loc[:, 'vol']
AMOUNT = df.loc[:, 'amount']
ADJ = df.loc[:, 'adj_factor']
suspend = df.loc[:, 'suspend_type']

AVG = AMOUNT / VOL

AVG_hfq = np.log(AVG * ADJ)

r_d = AVG_hfq.shift(-2) - AVG_hfq.shift(-1)
r_w = AVG_hfq.shift(-6) - AVG_hfq.shift(-1)
r_m = AVG_hfq.shift(-21) - AVG_hfq.shift(-1)

r_d_p = tools.standardize(tools.winsorize(r_d))
r_w_p = tools.standardize(tools.winsorize(r_w))
r_m_p = tools.standardize(tools.winsorize(r_m))

yiziban = (HIGH == LOW).astype(int)
suspend = suspend.copy()
suspend[suspend.notna()] = 1
suspend.fillna(0, inplace=True)

days_new = 250
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
is_trade.replace(0, np.nan, inplace=True)
is_trade = is_trade.shift(-1)
is_trade.iloc[-1, :] = 1
df = pd.concat({'R_DAILY':r_d, 
                'R_WEEKLY':r_w, 
                'R_MONTHLY':r_m, 
                'PREPROCESSED_R_DAILY':r_d_p, 
                'PREPROCESSED_R_WEEKLY':r_w_p, 
                'PREPROCESSED_R_MONTHLY':r_m_p, 
                'IS_TRADE':is_trade}, axis=1)
df = df.stack()

df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/label?charset=utf8")
df.to_sql('tdailylabel', engine, schema='label', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)
