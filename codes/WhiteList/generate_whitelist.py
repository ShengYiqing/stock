
import datetime
import os
import sys
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
from sqlalchemy import create_engine

factors = ['mc', 'amount', 
           'fjzc', 'fjyzc', 
           'fyysr', 'fyysrxj', 
           'fml', 'fmlxj', 'fhxlr', 'fzhsy', 'fjyxjll', 
           # 'expectedyysr', 'expectedjlr', 'analystcoverage',
           # 'quality',
           ]

factors_dic = {
    'gm': ['mc', 'amount'], 
    'zc': ['fjzc', 'fjyzc'], 
    'yy': ['fyysr', 'fyysrxj'], 
    'yl': ['fml', 'fmlxj', 'fhxlr', 'fzhsy', 'fjyxjll'], 
    # 'yq': ['expectedyysr', 'expectedjlr', 'analystcoverage'], 
    # 'zl': ['quality'],
    }

end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
start_date = '20100101'

trade_dates = tools.get_trade_cal(start_date, end_date)

sql = tools.generate_sql_y_x(factors, start_date, end_date, white_threshold=None, is_trade=False, white_ind=False, factor_value_type='factor_value')
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine)
df.replace(0, np.nan, inplace=True)
sql_s_i = """
select stock_code, l1_name ind_1, l2_name ind_2, l3_name ind_3 from indsw.tindsw
where (not isnull(l1_name)) and (not isnull(l2_name)) and (not isnull(l3_name))
"""
df_s_i = pd.read_sql(sql_s_i, engine).set_index('stock_code').loc[:, ['ind_1', 'ind_2', 'ind_3']]
df.loc[:, 'ind_1'] = [df_s_i.loc[i, 'ind_1'] if i in df_s_i.index else 'iuk' for i in df.loc[:, 'stock_code']]
df.loc[:, 'ind_2'] = [df_s_i.loc[i, 'ind_2'] if i in df_s_i.index else 'iuk' for i in df.loc[:, 'stock_code']]
df.loc[:, 'ind_3'] = [df_s_i.loc[i, 'ind_3'] if i in df_s_i.index else 'iuk' for i in df.loc[:, 'stock_code']]

df = df.set_index(['trade_date', 'stock_code', 'ind_1', 'ind_2', 'ind_3'])

for factor in factors:
    df.loc[:, factor] = df.loc[:, factor].groupby(['trade_date']).apply(lambda x:x.rank()/x.notna().sum())

for factor_k in factors_dic.keys():
    factor_v = factors_dic[factor_k]
    df.loc[:, factor_k] = df.loc[:, factor_v].mean(1)

factor_ks = list(factors_dic.keys())

df.loc[:, 'score'] = (
    8 * df.loc[:, factor_ks].mean(1).groupby(['trade_date']).apply(lambda x:x.rank()/x.notna().sum()) + 
    4 * df.loc[:, factor_ks].mean(1).groupby(['trade_date', 'ind_1']).apply(lambda x:x.rank()/x.notna().sum()) + 
    2 * df.loc[:, factor_ks].mean(1).groupby(['trade_date', 'ind_2']).apply(lambda x:x.rank()/x.notna().sum()) + 
    1 * df.loc[:, factor_ks].mean(1).groupby(['trade_date', 'ind_3']).apply(lambda x:x.rank()/x.notna().sum())
    ).groupby(['trade_date']).apply(lambda x:x.rank()/x.notna().sum())
df.loc[:, 'top_1'] = df.loc[:, 'score'].groupby(['trade_date', 'ind_1']).apply(lambda x:(x.rank()>(x.notna().sum()-5)))
df.loc[:, 'top_2'] = df.loc[:, 'score'].groupby(['trade_date', 'ind_2']).apply(lambda x:(x.rank()>(x.notna().sum()-1)))
df.loc[:, 'top_3'] = df.loc[:, 'score'].groupby(['trade_date', 'ind_3']).apply(lambda x:(x.rank()>(x.notna().sum()-1)))
df.loc[:, 'top'] = (df.loc[:, 'top_1'] | df.loc[:, 'top_2'] | df.loc[:, 'top_3']).astype(int)
df = df.loc[:, ['score', 'top'] + factor_ks]
df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
df = df.reset_index(['ind_1', 'ind_2', 'ind_3'], drop=True)
df.to_sql('tdailywhitelist', engine, schema='whitelist', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
if __name__ == '__main__':
    pass