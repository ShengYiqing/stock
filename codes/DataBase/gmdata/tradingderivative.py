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
    # start_date = (datetime.datetime.today() - datetime.timedelta(5)).strftime('%Y-%m-%d')
    # start_date ='2000-01-01'
    
    fields = 'DY,EV,EVEBITDA,EVPS,LYDY,NEGOTIABLEMV,PB,PCLFY,PCTTM,PELFY,PELFYNPAAEI,PEMRQ,PEMRQNPAAEI,PETTM,PETTMNPAAEI,PSLFY,PSMRQ,PSTTM,TCLOSE,TOTMKTCAP,TRADEDATE,TURNRATE,TOTAL_SHARE,FLOW_SHARE'
    
    # for stock in stock_list:
    df = query.get_fundamentals_n(table='trading_derivative_indicator', symbols=','.join(stock_list), count=5, end_date=end_date, fields=fields, df=True)
    if len(df) > 0:
        df.loc[:, 'stock_code'] = [sc.split('.')[1] for sc in df.loc[:, 'symbol']]
        if 'pub_date' in df.columns:
            del df['pub_date']
            
        df.rename(columns={'end_date': 'trade_date'}, inplace=True)

        df.loc[:, 'trade_date'] = [i.strftime('%Y%m%d') for i in df.trade_date]
    
        df.loc[:, 'REC_CREATOR'] = 'gm'
        df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        df.loc[:, 'REC_REVISOR'] = 'gm'
        df.loc[:, 'REC_REVISE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        
        df.to_sql('tgmtradingderivative', engine, schema='gmdata', index=False, chunksize=1000, if_exists='append', method=tools.mysql_replace_into)
