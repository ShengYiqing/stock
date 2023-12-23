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

sql = tools.generate_sql_y_x([], start_date, end_date)
df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
r_a = df.r_a.unstack()
r_o = df.r_o.unstack()
r_c = df.r_c.unstack()
mc = np.log(df.style_mc).unstack()
bp = np.log(1 / df.style_pb).unstack()

mc_ic_a = mc.corrwith(r_a, axis=1, method='spearman')
mc_ic_o = mc.corrwith(r_o, axis=1, method='spearman')
mc_ic_c = mc.corrwith(r_c, axis=1, method='spearman')
bp_ic_a = bp.corrwith(r_a, axis=1, method='spearman')
bp_ic_o = bp.corrwith(r_o, axis=1, method='spearman')
bp_ic_c = bp.corrwith(r_c, axis=1, method='spearman')

df_mc = pd.concat({'ic_a':mc_ic_a,  
                   'ic_o':mc_ic_o, 
                   'ic_c':mc_ic_c, 
                   }, axis=1)
df_mc = df_mc.loc[df_mc.index>=start_date]
df_mc.loc[:, 'style_code'] = 'mc'
df_mc.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
df_mc.to_sql('tstyledailylabel', engine, schema='label', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)

df_bp = pd.concat({'ic_a':bp_ic_a,  
                   'ic_o':bp_ic_o, 
                   'ic_c':bp_ic_c, 
                   }, axis=1)
df_bp = df_bp.loc[df_bp.index>=start_date]
df_bp.loc[:, 'style_code'] = 'bp'
df_bp.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
df_bp.to_sql('tstyledailylabel', engine, schema='label', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)
