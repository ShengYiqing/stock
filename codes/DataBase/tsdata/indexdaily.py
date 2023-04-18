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

end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')

# start_date = '20000101'

df = tools.download_tushare(pro=pro, api_name='index_daily', ts_code='000300.SH', start_date=start_date, end_date=end_date, fields='ts_code, trade_date, open, high, low, close, pre_close, vol, amount')
if len(df) > 0:
    df.rename({'ts_code':'index_code'}, axis=1, inplace=True)
    df.loc[:, 'index_name'] = '沪深300'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
     
    df.to_sql('ttsindexdaily', engine, schema='tsdata', index=False, if_exists='append', method=tools.mysql_replace_into)
