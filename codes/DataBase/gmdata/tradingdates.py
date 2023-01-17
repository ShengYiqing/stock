# coding=utf-8
from __future__ import print_function, absolute_import
import gm.api.basic as basic
import gm.api.query as query
import Config
import sys
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools

import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import multiprocessing as mp
import os
from sqlalchemy import create_engine


if __name__ == '__main__':

    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/gmdata?charset=utf8")
    f = open("%s/gm.txt"%gc.PROJECT_PATH)
    s = f.read()
    f.close()
    token = s
    basic.set_token(token)

    start_date='2000-01-01'
    end_date = (datetime.datetime.today() + datetime.timedelta(800)).strftime('%Y-%m-%d')

    trade_dates = query.get_trading_dates(exchange='SHSE', start_date=start_date, end_date=end_date)
    df1 = DataFrame({'trade_date':trade_dates})
    df1.loc[:, 'exchange_code'] = 'SHSE'
    
    trade_dates = query.get_trading_dates(exchange='SZSE', start_date=start_date, end_date=end_date)
    df2 = DataFrame({'trade_date':trade_dates})
    df2.loc[:, 'exchange_code'] = 'SZSE'
    
    df = pd.concat([df1, df2], axis=0)
    
    df.loc[:, 'REC_CREATOR'] = 'gm'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.loc[:, 'REC_REVISOR'] = 'gm'
    df.loc[:, 'REC_REVISE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    
    df.to_sql('tgmtradecal', engine, schema='gmdata', index=False, if_exists='append', method=tools.mysql_replace_into)
