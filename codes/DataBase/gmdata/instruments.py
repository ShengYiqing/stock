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

    sql = """
    select symbol from tgminstrumentinfos
    """
    stock_list = list(pd.read_sql(sql, engine).loc[:, 'symbol'])
        
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(5)).strftime('%Y-%m-%d')
    # end_date = '2022-04-11'
    # start_date ='2022-04-11'
    
    fields = 'symbol, trade_date, sec_level, pre_close, adj_factor, is_suspended'
        
    df = query.get_history_instruments(symbols=stock_list, start_date=start_date, end_date=end_date, fields=fields, df=True)
    if len(df) > 0:
        df.loc[:, 'stock_code'] = [sc.split('.')[1] for sc in df.loc[:, 'symbol']]
        df.loc[:, 'trade_date'] = [i.strftime('%Y%m%d') for i in df.trade_date]
    
        df.loc[:, 'REC_CREATOR'] = 'gm'
        df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        df.loc[:, 'REC_REVISOR'] = 'gm'
        df.loc[:, 'REC_REVISE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        
        df.to_sql('tgminstruments', engine, schema='gmdata', index=False, if_exists='append', method=tools.mysql_replace_into)
