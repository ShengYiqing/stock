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

# start_date = '20100101'

sql_trade_cal = """
select distinct cal_date from ttstradecal where is_open = 1
"""

trade_cal = list(pd.read_sql(sql_trade_cal, engine).loc[:, 'cal_date'])
trade_cal = list(filter(lambda x:(x>=start_date) & (x<=end_date), trade_cal))

for trade_date in trade_cal:
    # print(trade_date)
    df = tools.download_tushare(pro=pro, api_name='stk_limit', trade_date=trade_date)
    
    if len(df) > 0:
        df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]
        df.loc[:, 'REC_CREATOR'] = 'ts'
        df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        df.to_sql('ttsstklimit', engine, schema='tsdata', index=False, if_exists='append', method=tools.mysql_replace_into)
