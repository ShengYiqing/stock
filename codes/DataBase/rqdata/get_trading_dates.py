import os
import sys
import time
import datetime
import config
sys.path.append(config.DIR_GC)
import Global_Config as gc
import tools
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import rqdatac
from sqlalchemy import create_engine

if __name__ == '__main__':
    engine = create_engine("mysql+pymysql://root:chenYiYi0517@127.0.0.1:3306/rqdata?charset=utf8")
     
    rqdatac.init()
    end_date = (datetime.datetime.today() + datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    # start_date = '20100101'
    trade_cal = rqdatac.get_trading_dates(start_date, end_date, market='cn')
    df_trade_cal = DataFrame({'TRADE_DATE':[trade_date.strftime('%Y%m%d') for trade_date in trade_cal]})
    df_trade_cal.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df_trade_cal.to_sql('trqtradecal', engine, schema='rqdata', if_exists='append', index=False, chunksize=10000, method=tools.mysql_replace_into)
    
    # df_all_instrument = rqdatac.all_instruments(type='Future', market='cn', date=None)
    # df_all_instrument.to_csv('%s/all_instruments/all_instruments.csv'%gc.DIR_RQDATA)
    
    # underlying_symbols = list(set(list(df_all_instrument.underlying_symbol)))
    # for underlying_symbol in underlying_symbols:
    #     df_dom_list = rqdatac.futures.get_dominant(underlying_symbol)
    #     if df_dom_list is not None:
    #         df_dom_list.to_csv('%s/dom_list/%s.csv'%(gc.DIR_RQDATA, underlying_symbol))
    #     else:
    #         print(underlying_symbol)