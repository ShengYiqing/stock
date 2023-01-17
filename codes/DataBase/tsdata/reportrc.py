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
# start_date = '20220101'
# end_date = '20230101'
df = tools.download_tushare(pro=pro, api_name='report_rc', limit=2000, start_date=start_date, end_date=end_date)
df.dropna(subset='quarter', inplace=True)
if len(df) > 0:
    df.loc[df.loc[:, 'author_name'].isna(), 'author_name'] = '无'
    df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.to_sql('ttsreportrc', engine, schema='tsdata', index=False, chunksize=10000, if_exists='append', method=tools.mysql_replace_into)
