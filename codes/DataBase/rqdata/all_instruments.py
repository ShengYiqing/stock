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
    end_date = (datetime.datetime.today() + datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    today = datetime.datetime.today().strftime('%Y%m%d')
    # df_all_instrument = rqdatac.all_instruments(type='Future', market='cn', date=None)
    df_all_instrument = rqdatac.all_instruments(type='Future', market='cn', date=today)
    
    df_all_instrument.loc[:, 'maturity_date'] = [str(s).replace('-', '') for s in df_all_instrument.loc[:, 'maturity_date']]
    df_all_instrument.loc[:, 'listed_date'] = [str(s).replace('-', '') for s in df_all_instrument.loc[:, 'listed_date']]
    df_all_instrument.loc[:, 'de_listed_date'] = [str(s).replace('-', '') for s in df_all_instrument.loc[:, 'de_listed_date']]
    df_all_instrument.loc[:, 'start_delivery_date'] = [str(s).replace('-', '') for s in df_all_instrument.loc[:, 'start_delivery_date']]
    df_all_instrument.loc[:, 'end_delivery_date'] = [str(s).replace('-', '') for s in df_all_instrument.loc[:, 'end_delivery_date']]
    df_all_instrument.loc[:, 'future_code'] = df_all_instrument.loc[:, 'underlying_symbol']
    
    cols = ['order_book_id', 'underlying_symbol', 'market_tplus', 'symbol',
           'margin_rate', 'maturity_date', 'type', 'trading_code', 'exchange',
           'product', 'contract_multiplier', 'round_lot', 'trading_hours',
           'listed_date', 'industry_name', 'de_listed_date',
           'start_delivery_date', 'end_delivery_date', 'REC_CREATE_TIME', 'future_code']
    df_all_instrument.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df_all_instrument.loc[:, cols].to_sql('trqinstruments', engine, schema='rqdata', if_exists='append', index=False, chunksize=10000, method=tools.mysql_replace_into)
    
    # underlying_symbols = list(set(list(df_all_instrument.underlying_symbol)))
    # for underlying_symbol in underlying_symbols:
    #     df_dom_list = rqdatac.futures.get_dominant(underlying_symbol)
    #     if df_dom_list is not None:
    #         df_dom_list.to_csv('%s/dom_list/%s.csv'%(gc.DIR_RQDATA, underlying_symbol))
    #     else:
    #         print(underlying_symbol)