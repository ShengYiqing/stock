import os
import sys
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
from gm.api import *
import gm.api.basic as basic
import gm.api.query as query
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

end_date = '20190101'
start_date = '20170811'

end_date = '20210101'
start_date = '20201001'

trade_dates = tools.get_trade_cal(start_date, end_date)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306")

for trade_date in trade_dates:
    try:
        sql = """
            CREATE TABLE `mindata`.`tmindata%s` (
          `REC_CREATE_TIME` VARCHAR(14) NULL DEFAULT ' ',
          `STOCK_CODE` VARCHAR(20) NOT NULL DEFAULT ' ',
          `TRADE_DATE` VARCHAR(8) NOT NULL DEFAULT ' ',
          `TRADE_TIME` VARCHAR(6) NOT NULL DEFAULT ' ',
          `OPEN` DOUBLE NULL,
          `HIGH` DOUBLE NULL,
          `LOW` DOUBLE NULL,
          `CLOSE` DOUBLE NULL,
          `SPREAD` DOUBLE NULL,
          `VOL` DOUBLE NULL,
          `AMOUNT` DOUBLE NULL,
          `IMBALANCE` DOUBLE NULL,
          PRIMARY KEY (`STOCK_CODE`, `TRADE_DATE`, `TRADE_TIME`))
            """%trade_date
        with engine.connect() as con:
            con.execute(sql)
    except:
        pass
    f = open("%s/gm.txt"%gc.PROJECT_PATH)
    s = f.read()
    f.close()
    token = s
    basic.set_token(token)
    stocks = get_instruments(sec_types=1, fields='symbol', skip_suspended=True, skip_st=False, df=True)
    stocks = list(stocks.iloc[:, 0])
    stocks = list(filter(lambda x:x[5] == '0' or x[5] == '3' or x[5] == '4' or x[5] == '6' or x[5] == '8', stocks))
    m = len(stocks)
    n = 0
    # df_list = []
    while n < m:
        print(trade_date, n)
        stocks_tmp = stocks[n:min(n+130, m)]
        df = history(symbol=stocks_tmp, frequency='60s', 
                     fields='symbol, open, high, low, close, volume, amount, bob', 
                     start_time=datetime.datetime.strptime(trade_date, '%Y%m%d'), 
                     end_time=(datetime.datetime.strptime(trade_date, '%Y%m%d')+datetime.timedelta(1)), 
                     df=True)
        # df_list.append(df_tmp)
        n = n+130
    # df = pd.concat(df_list, axis=0, ignore_index=True)
        if len(df) > 0:
            df.loc[:, 'stock_code'] = [i.split('.')[1] for i in df.symbol]
            df.loc[:, 'rec_create_time'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
            df.loc[:, 'trade_date'] = [i.strftime('%Y%m%d') for i in df.bob]
            df.loc[:, 'trade_time'] = [i.strftime('%H%M%S') for i in df.bob]
            df.loc[:, 'vol'] = df.volume
            df = df.loc[:, ['rec_create_time', 
                            'trade_date', 'trade_time', 'stock_code', 
                            'open', 'high', 'low', 'close', 
                            'vol', 'amount']]
            df.to_sql('tmindata%s'%trade_date, engine, schema='mindata', index=False, if_exists='append', method=tools.mysql_replace_into)
    