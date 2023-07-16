import tushare as ts
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
import sqlalchemy as sa
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

pro = ts.pro_api()

sql = """
select ts_code from ttsstockbasic
"""
stock_list = list(pd.read_sql(sql, engine).ts_code)
n = len(stock_list)
stocks_per_iter = 5
iters = int(n / stocks_per_iter) + 1
for i in range(iters):
    time.sleep(0.2)
    start = i * stocks_per_iter
    end = min(start+stocks_per_iter, n)
    if start >= end:
        break
    stocks = ','.join(stock_list[start:end])
    df = tools.download_tushare(pro=pro, api_name='stk_managers', ts_code=stocks, fields='ts_code,ann_date,name,gender,lev,title,edu,national,birthday,begin_date,end_date')
    if len(df) > 0:
        df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]

        df.loc[:, 'REC_CREATOR'] = 'ts'
        df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

        df.to_sql('ttsstkmanagers', engine, schema='tsdata', index=False, if_exists='append', method=tools.mysql_replace_into)
