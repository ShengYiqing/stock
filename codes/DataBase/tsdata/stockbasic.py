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

df1 = tools.download_tushare(pro=pro, api_name='stock_basic', list_status='D', fields='ts_code,name,area,industry,market,exchange,list_status,list_date,delist_date,is_hs')
df2 = tools.download_tushare(pro=pro, api_name='stock_basic', list_status='L', fields='ts_code,name,area,industry,market,exchange,list_status,list_date,delist_date,is_hs')
df3 = tools.download_tushare(pro=pro, api_name='stock_basic', list_status='P', fields='ts_code,name,area,industry,market,exchange,list_status,list_date,delist_date,is_hs')
df = pd.concat([df1, df2, df3], axis=0)

if len(df) > 0:
    df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]
    
    df.loc[:, 'REC_CREATOR'] = 'ts'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.loc[:, 'REC_REVISOR'] = 'ts'
    df.loc[:, 'REC_REVISE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    
    df.to_sql('ttsstockbasic', engine, schema='tsdata', index=False, if_exists='append', method=tools.mysql_replace_into)
