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
(t3.total_mv / t3.total_share * t3.free_share) mc, (1 / t3.pb) bp
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
mc = np.log(df.loc[:, 'mc'])
bp = np.log(df.loc[:, 'bp']).replace(-np.inf, np.nan)
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

beta = (r.ewm(halflife=20, min_periods=60).corr(r_m) * r.ewm(halflife=20, min_periods=60).std()).div(r_m.ewm(halflife=20, min_periods=60).std(), axis=0)

rank_beta = beta.rank(axis=1, pct=True)
rank_mc = mc.rank(axis=1, pct=True)
rank_bp = bp.rank(axis=1, pct=True)
df = pd.concat({'beta':beta,  
                'mc':mc,  
                'bp':bp,  
                'rank_beta':rank_beta,  
                'rank_mc':rank_mc,  
                'rank_bp':rank_bp,  
                }, axis=1)
df = df.loc[df.index>=start_date]
df = df.stack()

df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/label?charset=utf8")
df.to_sql('tdailystyle', engine, schema='style', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)
