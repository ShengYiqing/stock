#!/usr/bin/env python
# coding: utf-8

#%%
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import tushare as ts

import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR

#%%
def generate_factor(start_date, end_date):
    start_date_sql = tools.trade_date_shift(start_date, 1)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

    sql = """
    select t1.trade_date, t1.stock_code, 
    t1.open, t1.close, t1.high, t1.low, t1.vol, t1.amount,
    t2.adj_factor, 
    tud.up_limit, tud.down_limit 
    from ttsdaily t1
    left join ttsadjfactor t2
    on t1.stock_code = t2.stock_code
    and t1.trade_date = t2.trade_date
    left join ttsstklimit tud
    on t1.stock_code = tud.stock_code
    and t1.trade_date = tud.trade_date
    where t1.trade_date >= {start_date}
    and t1.trade_date <= {end_date}
    """
    sql = sql.format(start_date=start_date_sql, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    
    o = np.log(df.loc[:, 'open'] * df.loc[:, 'adj_factor']).unstack()
    c = np.log(df.loc[:, 'close'] * df.loc[:, 'adj_factor']).unstack()
    r = c.diff()
    h = np.log(df.loc[:, 'high'] * df.loc[:, 'adj_factor']).unstack()
    l = np.log(df.loc[:, 'low'] * df.loc[:, 'adj_factor']).unstack()
    da = np.log(df.loc[:, 'amount']).unstack().diff()
    avg = np.log(10 * df.loc[:, 'amount'] / df.loc[:, 'vol'] * df.loc[:, 'adj_factor']).unstack()
    
    
    formulas_dic = {
        1: '-(c-avg)', 
        2: '-(c-o)', 
        3: '-(c-l)/(h-l)', 
        4: '(o-l)/(h-l)', 
        5: '-(c-h)', 
        6: '-(c-l)', 
        7: '-(h-o)', 
        8: '-(l-o)', 
        9: '-(c-l)/(h-l) * (1-(o-l)/(h-l))', +++
        10: '(h-o) * (h-c)', 
        11: '-(l-o) * (l-c)', 
        12: '-(h-o) * (c-l)', 
        13: '(l-o) * (c-h)', 
        14: '-(h-o) * (c-l)/(h-l)', 
        15: '(l-o) * (c-l)/(h-l)', 
        16: '-da * (c-o)', 
        17: '-da * r',
        }
    for k in formulas_dic.keys():
        print(k)
        try:
            sql = """
            CREATE TABLE `factor`.`tfactoralpha1d%s` (
              `REC_CREATE_TIME` VARCHAR(14) NULL,
              `TRADE_DATE` VARCHAR(8) NOT NULL,
              `STOCK_CODE` VARCHAR(18) NOT NULL,
              `FACTOR_VALUE` DOUBLE NULL,
              PRIMARY KEY (`TRADE_DATE`, `STOCK_CODE`))
            """%k
            with engine.connect() as con:
                con.execute(sql)
        except:
            pass
        exec('df = %s'%formulas_dic[k])
        df = df.loc[df.index>=start_date]
        df.replace(np.inf, np.nan, inplace=True)
        df.replace(-np.inf, np.nan, inplace=True)
        df.index.name = 'trade_date'
        df.columns.name = 'stock_code'
        df = DataFrame({'factor_value':df.stack()})
        df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        df.to_sql('tfactoralpha1d%s'%k, engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)