import tushare as ts
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import Global_Config as gc
import tools
import sqlalchemy as sa
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

pro = ts.pro_api()

end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
# start_date = '19800101'

df = tools.download_tushare(pro=pro, api_name='new_share', limit=2000, start_date=start_date, end_date=end_date, fields='ts_code, sub_code, name, ipo_date, issue_date, amount, market_amount, price, pe, limit_amount, funds, ballot')

df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]
df.loc[:, 'REC_CREATOR'] = 'ts'
df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
df.loc[:, 'REC_REVISOR'] = 'ts'
df.loc[:, 'REC_REVISE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    
df.to_sql('ttsnewshare', engine, schema='tsdata', index=False, if_exists='append', method=tools.mysql_replace_into)
