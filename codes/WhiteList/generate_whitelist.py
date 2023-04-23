
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

factors = ['liquidity', 'mc']

end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
# start_date = '20100101'

trade_dates = tools.get_trade_cal(start_date, end_date)

factor_value_type_dic = {'liquidity': 'factor_value', 
                         'mc': 'factor_value', 
                         }
sql = tools.generate_sql_y_x(factors, start_date, end_date, is_industry=False, is_white=False, is_trade=False, factor_value_type_dic=factor_value_type_dic)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine)
df = df.set_index(['trade_date', 'stock_code']).loc[:, factors].groupby(['trade_date']).rank(pct=True)

df.loc[:, 'white'] = ((df.mc>0.05) & (df.liquidity>0.05)).astype(int)
df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
df.to_sql('tdailywhitelist', engine, schema='whitelist', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)
if __name__ == '__main__':
    pass