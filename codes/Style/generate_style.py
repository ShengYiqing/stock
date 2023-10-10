import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import Global_Config as gc
import tools
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
# start_date = '20100101'

start_date_sql = tools.trade_date_shift(start_date, 250)

sql = """
select 
t1.stock_code, t1.trade_date, 
t1.close, t2.adj_factor, 
(t3.total_mv / t3.total_share * t3.free_share) mc, 
(t3.total_mv / t3.pb) b, (t3.total_mv / t3.pe_ttm) e, (t3.pb / t3.pe_ttm) roe, 
t3.pb, t3.pe_ttm pe, t3.dv_ttm dv
from tsdata.ttsdaily t1
left join tsdata.ttsadjfactor t2
on t1.trade_date = t2.trade_date
and t1.stock_code = t2.stock_code
left join tsdata.ttsdailybasic t3
on t1.trade_date = t3.trade_date
and t1.stock_code = t3.stock_code
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)

df = pd.read_sql(sql, engine)
df = df.set_index(['trade_date', 'stock_code']).unstack()
c = df.loc[:, 'close']
mc = df.loc[:, 'mc']
b = df.loc[:, 'b']
e = df.loc[:, 'e']
roe = df.loc[:, 'roe']
pb = df.loc[:, 'pb']
pe = df.loc[:, 'pe']
dv = df.loc[:, 'dv']
adj = df.loc[:, 'adj_factor']
close_hfq = np.log(c * adj)
r = close_hfq.diff()

sql = """
select trade_date, close from ttsindexdaily
where trade_date >= {start_date}
and trade_date <= {end_date}
and index_name = '沪深300'
""".format(start_date=start_date_sql, end_date=end_date)
close_m = pd.read_sql(sql, engine).set_index('trade_date').loc[:, 'close']
r_m = np.log(close_m).diff()

beta = (r.ewm(halflife=60, min_periods=60).corr(r_m) * r.ewm(halflife=60, min_periods=60).std()).div(r_m.ewm(halflife=60, min_periods=60).std(), axis=0)

rank_beta = beta.rank(axis=1, pct=True)
rank_mc = mc.rank(axis=1, pct=True)
rank_b = b.rank(axis=1, pct=True)
rank_e = e.rank(axis=1, pct=True)
rank_roe = roe.rank(axis=1, pct=True)
rank_pb = pb.rank(axis=1, pct=True)
rank_pe = pe.rank(axis=1, pct=True)
rank_dv = dv.rank(axis=1, pct=True)
df = pd.concat({'beta':beta,  
                'mc':mc,
                'b':b,
                'e':e,
                'roe':roe, 
                'pb':pb, 
                'pe':pe, 
                'dv':dv, 
                'rank_beta':rank_beta, 
                'rank_mc':rank_mc,  
                'rank_b':rank_b,
                'rank_e':rank_e,
                'rank_roe':rank_roe, 
                'rank_pb':rank_pb, 
                'rank_pe':rank_pe, 
                'rank_dv':rank_dv, 
                }, axis=1)
df = df.loc[df.index>=start_date]
df = df.stack()

df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/label?charset=utf8")
df.to_sql('tdailystyle', engine, schema='style', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)
