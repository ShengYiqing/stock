# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import gm.api.basic as basic
import gm.api.query as query
import tushare as ts
import datetime
import time
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sys
import os
import multiprocessing as mp
import pickle
import Config

sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools

def f(date, stock):
    try:
        if os.path.exists("D:/stock/DataBase/StockSnapshootData/%s/%s.csv"%(date, stock)):
            return
        f = open("%s/gm.txt"%gc.PROJECT_PATH)
        s = f.read()
        f.close()
        token = s
        basic.set_token(token)
        
        history_data = history(symbol=stock, fields='quotes, created_at, price, last_volume, last_amount, trade_type', frequency='tick', start_time=datetime.datetime.strptime(date, '%Y%m%d'),  end_time=(datetime.datetime.strptime(date, '%Y%m%d')+datetime.timedelta(1)), df=True)
        
        if len(history_data) == 0:
            return
        history_data.loc[:, 'created_at'] = [timestamp.tz_localize(tz=None) for timestamp in history_data.loc[:, 'created_at']]
        history_data = history_data.set_index('created_at')
        
        history_data_1 = history_data.loc[history_data.index < '%s150100'%date, :].copy()
        history_data_2 = history_data.loc[history_data.index > '%s150100'%date, :].copy()
        
        history_data_1.loc[:, 'bid_price_1'] = [i[0]['bid_p'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'bid_price_2'] = [i[1]['bid_p'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'bid_price_3'] = [i[2]['bid_p'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'bid_price_4'] = [i[3]['bid_p'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'bid_price_5'] = [i[4]['bid_p'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_price_1'] = [i[0]['ask_p'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_price_2'] = [i[1]['ask_p'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_price_3'] = [i[2]['ask_p'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_price_4'] = [i[3]['ask_p'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_price_5'] = [i[4]['ask_p'] for i in history_data_1.quotes]
        
        history_data_1.loc[:, 'bid_vol_1'] = [i[0]['bid_v'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'bid_vol_2'] = [i[1]['bid_v'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'bid_vol_3'] = [i[2]['bid_v'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'bid_vol_4'] = [i[3]['bid_v'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'bid_vol_5'] = [i[4]['bid_v'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_vol_1'] = [i[0]['ask_v'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_vol_2'] = [i[1]['ask_v'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_vol_3'] = [i[2]['ask_v'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_vol_4'] = [i[3]['ask_v'] for i in history_data_1.quotes]
        history_data_1.loc[:, 'ask_vol_5'] = [i[4]['ask_v'] for i in history_data_1.quotes]
        history_data_1.drop(labels='quotes', axis=1, inplace=True)
    
        history_data_2.loc[:, 'bid_price_1'] = [i[0]['bid_p'] for i in history_data_2.quotes]
        history_data_2.loc[:, 'ask_price_1'] = [i[0]['ask_p'] for i in history_data_2.quotes]
        
        history_data_2.loc[:, 'bid_vol_1'] = [i[0]['bid_v'] for i in history_data_2.quotes]
        history_data_2.loc[:, 'ask_vol_1'] = [i[0]['ask_v'] for i in history_data_2.quotes]
        history_data_2.drop(labels='quotes', axis=1, inplace=True)
    
        history_data = pd.concat([history_data_1, history_data_2], axis=0)
        history_data.to_csv("D:/stock/DataBase/StockSnapshootData/%s/%s.csv"%(date, stock))
        print(date, stock, ' download')
    except:
        print(date, stock, ' fail')
if __name__ == '__main__':
    date = datetime.datetime.today().strftime('%Y%m%d')
    # date = '20230411'
    start_date = date
    end_date = date
    trade_cal = tools.get_trade_cal(start_date=date, end_date=date)
    if len(trade_cal) == 0:
        sys.exit()
    file = open("%s/gm.txt"%gc.PROJECT_PATH)
    s = file.read()
    file.close()
    token = s
    basic.set_token(token)
    
    
    stocks = get_instruments(sec_types=1, fields='symbol', skip_suspended=False, skip_st=False, df=True)
    stocks = list(stocks.iloc[:, 0])
    stocks = list(filter(lambda x:x[5] == '0' or x[5] == '3' or x[5] == '6' or x[5] == '8', stocks))

    pro = ts.pro_api()
    #获取日期
    df_dates = pro.trade_cal(exchange='SZSE', start_date=start_date, end_date=end_date)
    dates = df_dates.cal_date[df_dates.is_open==1]
    
    #取数写入
    pool = mp.Pool(2)

    for date in dates:
        pool.apply_async(func=f, args=(date, stocks))
        if not os.path.exists('D:/stock/DataBase/StockSnapshootData/%s'%date):
            os.mkdir('D:/stock/DataBase/StockSnapshootData/%s'%date)
        for stock in stocks:
            pool.apply_async(func=f, args=(date, stock))
            # f(date, stock)
    pool.close()
    pool.join()