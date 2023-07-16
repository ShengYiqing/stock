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
start_date = '20120101'
end_date = '20230705'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql = """
select tl.trade_date, tl.stock_code, tl.r_d_a, tl.r_d_o, tl.r_d_c 
from label.tdailylabel tl
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
where tl.trade_date >= {start_date}
and tl.trade_date <= {end_date}
and tl.rank_mc > 0.9
and tl.rank_cmc > 0.382
and tl.rank_amount > 0.382
and tl.rank_price > 0.382
and tl.rank_revenue > 0.382
and ti.l3_name in {white_ind}
""".format(start_date=start_date, end_date=end_date, white_ind=tuple(gc.WHITE_INDUSTRY_LIST))
y_a = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_d_a.unstack()
y_o = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_d_o.unstack()
y_c = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_d_c.unstack()
y_c_s = y_c.shift(-1)
stock_codes = list(y_a.columns)
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
end_date = '20230705'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql = """
select tl.trade_date, tl.stock_code, tl.r_d_a, tl.r_d_o, tl.r_d_c 
from label.tdailylabel tl
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
where tl.trade_date >= {start_date}
and tl.trade_date <= {end_date}
and tl.rank_mc > 0.8
and tl.rank_cmc > 0.2
and tl.rank_amount > 0.2
and tl.rank_price > 0.2
and tl.rank_revenue > 0.2
and ti.l3_name in {white_ind}
""".format(start_date=start_date, end_date=end_date, white_ind=tuple(gc.WHITE_INDUSTRY_LIST))
y_a = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_d_a.unstack()
y_o = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_d_o.unstack()
y_c = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).r_d_c.unstack()
y_c_s = y_c.shift(-1)
stock_codes = list(y_a.columns)
#%%
n = 250
start_date_sql = tools.trade_date_shift(start_date, n+1)

sql = """
select t1.trade_date, t1.stock_code, 
t1.close, 
t2.adj_factor from tsdata.ttsdaily t1
left join tsdata.ttsadjfactor t2
on t1.stock_code = t2.stock_code
and t1.trade_date = t2.trade_date
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
df = pd.read_sql(sql, engine)

c = df.set_index(['trade_date', 'stock_code']).loc[:, 'close']

adj_factor = df.set_index(['trade_date', 'stock_code']).loc[:, 'adj_factor']
r = np.log(c * adj_factor).groupby('stock_code').diff()
r = r.unstack()

sql = """
select trade_date, close from tsdata.ttsindexdaily
where trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql = sql.format(start_date=start_date_sql, end_date=end_date)
close_m = pd.read_sql(sql, engine).set_index('trade_date').loc[:, 'close']
r_m = np.log(close_m).diff()

# x_1 = (r.ewm(halflife=5).corr(r_m) * r.ewm(halflife=5).std()).div(r_m.ewm(halflife=5).std(), axis=0)
# x_2 = (r.ewm(halflife=20).corr(r_m) * r.ewm(halflife=20).std()).div(r_m.ewm(halflife=20).std(), axis=0)
x = r.ewm(halflife=20).corr(r_m) * r.ewm(halflife=20).std()
# x = x_1 - x_2
x = x.replace(-np.inf, np.nan).replace(np.inf, np.nan)
# x = tools.neutralize(x)
x_ = DataFrame(x, index=y_a.index, columns=y_a.columns)
x_[y_a.isna()] = np.nan
tools.factor_analyse(x_, y_a, 7, 'ya')
# tools.factor_analyse(x_, y_o, 7, 'yo')
# tools.factor_analyse(x_, y_c, 7, 'yc')
# tools.factor_analyse(x_, y_c_s, 7, 'ycs')
