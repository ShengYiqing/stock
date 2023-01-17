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
stock_list = list(pd.read_sql(sql, engine).loc[:, 'ts_code'])
for stock in stock_list:
    df = tools.download_tushare(pro=pro, api_name='pledge_stat', ts_code=stock, fields=[
    "ts_code",
    "end_date",
    "pledge_count",
    "unrest_pledge",
    "rest_pledge",
    "total_share",
    "pledge_ratio"
    ])
    
    if len(df) > 0:
        df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]
    
        df.loc[:, 'REC_CREATOR'] = 'ts'
        df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        df.loc[:, 'REC_REVISOR'] = 'ts'
        df.loc[:, 'REC_REVISE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        
        df.to_sql('ttspledgestat', engine, schema='tsdata', index=False, if_exists='append', method=tools.mysql_replace_into)
