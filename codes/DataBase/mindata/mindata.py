import os
import sys
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import Global_Config as gc
import tools
import sqlalchemy as sa
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

end_date = datetime.datetime.today().strftime('%Y%m%d')
# end_date = '20230714'
start_date = end_date
# start_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')

# start_date = '20231030'
# end_date = '20231104'

sql_trade_cal = """
select distinct cal_date from ttstradecal where is_open = 1
"""

trade_cal = list(pd.read_sql(sql_trade_cal, engine).loc[:, 'cal_date'])
trade_cal = list(filter(lambda x:(x>=start_date) & (x<=end_date), trade_cal))

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/mindata?charset=utf8")

for trade_date in trade_cal:
    # try:
    #     sql = """
    #         CREATE TABLE `mindata`.`tmindata` (
    #       `REC_CREATE_TIME` VARCHAR(14) NULL DEFAULT ' ',
    #       `STOCK_CODE` VARCHAR(20) NOT NULL DEFAULT ' ',
    #       `TRADE_DATE` VARCHAR(8) NOT NULL DEFAULT ' ',
    #       `TRADE_TIME` VARCHAR(6) NOT NULL DEFAULT ' ',
    #       `OPEN` DOUBLE NULL,
    #       `HIGH` DOUBLE NULL,
    #       `LOW` DOUBLE NULL,
    #       `CLOSE` DOUBLE NULL,
    #       `SPREAD` DOUBLE NULL,
    #       `VOL` DOUBLE NULL,
    #       `AMOUNT` DOUBLE NULL,
    #       `IMBALANCE` DOUBLE NULL,
    #       PRIMARY KEY (`STOCK_CODE`, `TRADE_DATE`, `TRADE_TIME`))
    #         """
    #     with engine.connect() as con:
    #         con.execute(sql)
    # except:
    #     pass
    d = 'D:/stock/DataBase/StockSnapshootData/' + trade_date
    
    # d = 'E:/DataBase/StockSnapshootData/' + trade_date
    
    files = os.listdir(d)
    for file in files:
        print(trade_date, file)
        df = pd.read_csv(d+'/'+file, index_col=[0], parse_dates=[0]).loc[:, ['bid_price_1', 'ask_price_1', 'last_volume', 'last_amount']+['bid_vol_%s'%i for i in range(1, 6)]+['ask_vol_%s'%i for i in range(1, 6)]]
        df.loc[:, 'mid_price'] = df.loc[:, ['bid_price_1', 'ask_price_1']].replace(0, np.nan).mean(1).fillna(method='ffill')
        df.loc[:, 'spread'] = (np.log(df.loc[:, 'ask_price_1'].replace(0, np.nan)) - np.log(df.loc[:, 'bid_price_1'].replace(0, np.nan))).fillna(method='ffill')
        df.loc[:, 'bid_vol'] = df.loc[:, ['bid_vol_%s'%i for i in range(1, 6)]].sum(1).fillna(0)
        df.loc[:, 'ask_vol'] = df.loc[:, ['ask_vol_%s'%i for i in range(1, 6)]].sum(1).fillna(0)
        df.loc[:, 'imbalance'] = ((df.bid_vol - df.ask_vol) / (df.bid_vol + df.ask_vol)).replace(-np.inf, np.nan).replace(np.inf, np.nan).fillna(method='ffill')
        # df = df.resample('1min').agg({'mid_price':['first', 'max', 'min', 'last'], 'spread':'mean', 'last_volume':'sum', 'last_amount':'sum', 'imbalance':'mean'})
        df = df.resample('5min').agg({'mid_price':['first', 'max', 'min', 'last'], 'spread':'mean', 'last_volume':'sum', 'last_amount':'sum', 'imbalance':'mean'})
        
        df.columns = ['open', 'high', 'low', 'close', 'spread', 'vol', 'amount', 'imbalance']
        df.dropna(inplace=True)
        if len(df) > 0:
            df.loc[:, 'stock_code'] = file.split('.')[1]
            df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
            df.loc[:, 'trade_date'] = [i.strftime('%Y%m%d') for i in df.index]
            df.loc[:, 'trade_time'] = [i.strftime('%H%M%S') for i in df.index]
            
            df = df.loc[((df.index >= trade_date+'091500') & (df.index <= trade_date+'113000')) | ((df.index >= trade_date+'130000') & (df.index <= trade_date+'150000')), :]
            
            # df.to_sql('tmindata%s'%trade_date, engine, schema='mindata', index=False, if_exists='append', method=tools.mysql_replace_into)
            df.to_sql('tmindata', engine, schema='mindata', index=False, if_exists='append', method=tools.mysql_replace_into)
    