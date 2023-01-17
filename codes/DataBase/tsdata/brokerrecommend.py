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

#获取查询月份券商金股
n = 3
for i in range(n):
    time.sleep(15)
    month = (datetime.datetime.today() - datetime.timedelta(30*i)).strftime('%Y%m')
    df = tools.download_tushare(pro=pro, api_name='broker_recommend', month=month,  fields=[
    "month",
    "broker",
    "ts_code",
])
    if len(df) > 0:
        df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]
        df.loc[:, 'REC_CREATOR'] = 'ts'
        df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        df.loc[:, 'REC_REVISOR'] = 'ts'
        df.loc[:, 'REC_REVISE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        
        df.to_sql('ttsbrokerrecommend', engine, schema='tsdata', index=False, if_exists='append', method=tools.mysql_replace_into)
