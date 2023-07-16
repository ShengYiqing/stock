import os
import sys
import time
import datetime
import Global_Config as gc
import tools
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import rqdatac
from sqlalchemy import create_engine

if __name__ == '__main__':
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/rqdata?charset=utf8")
    
    rqdatac.init()
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')
    today = datetime.datetime.today().strftime('%Y%m%d')
    # df_all_instrument = rqdatac.all_instruments(type='Future', market='cn', date=None)
    df_min = rqdatac.get_price(order_book_ids='000001.XSHE', start_date=start_date, end_date=end_date, frequency='1m')
    
    df_all_instrument.loc[:, 'listed_date'] = [str(s).replace('-', '') for s in df_all_instrument.loc[:, 'listed_date']]
    df_all_instrument.loc[:, 'de_listed_date'] = [str(s).replace('-', '') for s in df_all_instrument.loc[:, 'de_listed_date']]
    df_all_instrument.loc[:, 'stock_code'] = df_all_instrument.loc[:, 'trading_code']
    
    df_all_instrument.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df_all_instrument.to_sql('trqinstruments', engine, schema='rqdata', if_exists='append', index=False, chunksize=10000, method=tools.mysql_replace_into)
    
    # underlying_symbols = list(set(list(df_all_instrument.underlying_symbol)))
    # for underlying_symbol in underlying_symbols:
    #     df_dom_list = rqdatac.futures.get_dominant(underlying_symbol)
    #     if df_dom_list is not None:
    #         df_dom_list.to_csv('%s/dom_list/%s.csv'%(gc.DIR_RQDATA, underlying_symbol))
    #     else:
    #         print(underlying_symbol)