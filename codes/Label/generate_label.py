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
start_date = (datetime.datetime.today() - datetime.timedelta(7)).strftime('%Y%m%d')
# start_date = '20100101'


sql = """
select t1.stock_code, t1.trade_date, 
       t1.open, t1.high, t1.low, t1.close, t1.vol, t1.amount, 
       t4.total_mv mc, t4.circ_mv circ_mc, 
       (t4.total_mv / t4.pb) b, (t4.total_mv / t4.ps_ttm) s, 
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
"""
sql = sql.format(start_date=start_date, end_date=end_date)

df = pd.read_sql(sql, engine)
df = df.set_index(['trade_date', 'stock_code']).unstack()
CLOSE = df.loc[:, 'close']
HIGH = df.loc[:, 'high']
LOW = df.loc[:, 'low']
VOL = df.loc[:, 'vol']
AMOUNT = df.loc[:, 'amount']
mc = df.loc[:, 'mc']
circ_mc = df.loc[:, 'circ_mc']
b = df.loc[:, 'b']
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

low_amount = (AMOUNT.rolling(250, min_periods=20).mean().rank(axis=1, pct=True) < 0.2).astype(int)
low_amount[AMOUNT.isna()] = 1

low_p = (CLOSE.rank(axis=1, pct=True) < 0.2).astype(int)
low_p[CLOSE.isna()] = 1

low_b = (b.rank(axis=1, pct=True) < 0.2).astype(int)
low_b[b.isna()] = 1

low_s = (s.rank(axis=1, pct=True) < 0.2).astype(int)
low_s[s.isna()] = 1

low_cmc = (circ_mc.rank(axis=1, pct=True) < 0.2).astype(int)
low_cmc[circ_mc.isna()] = 1

is_trade = yiziban + suspend + new + low_amount + low_p + low_b + low_s + low_cmc
is_trade[CLOSE.isna()] = 1
is_trade[is_trade>0] = 1

sql_ind = """
select stock_code, l3_name 
from indsw.tindsw
where l3_name in {l3}
""".format(l3=tuple(gc.WHITE_INDUSTRY_LIST))

# sql_ind = """
# select stock_code, l3_name 
# from indsw.tindsw
# """

df_ind = pd.read_sql(sql_ind, engine)
stock_ind_white = list(df_ind.stock_code)

is_trade.loc[:, list(set(is_trade.columns).difference(set(stock_ind_white)))] = 1.0

is_trade = 1 - is_trade

stock_ind_s = df_ind.set_index('stock_code').l3_name

def f(s):
    trade_date = s.name
    white_tmp = list(s.index[s==1])
    stock_ind_s_tmp = stock_ind_s[white_tmp]
    mc_tmp = mc.loc[trade_date, white_tmp]
    mc_ind_tmp = DataFrame({'mc':mc_tmp, 'ind':stock_ind_s_tmp})
    mc_ind_tmp = mc_ind_tmp.set_index('ind', append=True).groupby('ind').rank(ascending=False)
    # print(mc_ind_tmp)
    mc_ind_tmp = mc_ind_tmp.loc[mc_ind_tmp.mc<=5]
    # print(mc_ind_tmp)
    white_tmp = list(mc_ind_tmp.reset_index(-1).index)
    
    mc_tmp = mc.loc[trade_date, white_tmp].sort_values(ascending=False)
    white = mc_tmp.index[:min(180, len(mc_tmp))]
    ret = s.copy()
    ret.loc[:] = 0
    ret.loc[white] = 1
    return ret
is_white = is_trade.apply(f, axis=1)

df = pd.concat({'R_DAILY':r_d, 
                'R_WEEKLY':r_w, 
                'R_MONTHLY':r_m, 
                'IS_TRADE':is_trade,
                'IS_WHITE':is_white}, axis=1)
df = df.stack()

df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/label?charset=utf8")
df.to_sql('tdailylabel', engine, schema='label', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)
