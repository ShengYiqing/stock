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

df1 = tools.download_tushare(pro=pro, api_name='trade_cal', exchange='SSE', fields='exchange,cal_date,is_open,pretrade_date')
df2 = tools.download_tushare(pro=pro, api_name='trade_cal', exchange='SZSE', fields='exchange,cal_date,is_open,pretrade_date')
df = pd.concat([df1, df2], axis=0)

if len(df) > 0:
    df.rename({'exchange':'exchange_code'}, axis=1, inplace=True)
    
    df.loc[:, 'REC_CREATOR'] = 'ts'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.loc[:, 'REC_REVISOR'] = 'ts'
    df.loc[:, 'REC_REVISE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    
    df.to_sql('ttstradecal', engine, schema='tsdata', index=False, if_exists='append', method=tools.mysql_replace_into)
