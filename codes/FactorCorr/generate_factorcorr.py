# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:47:43 2021

@author: admin
"""

import os
import sys
import time
import datetime
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)

import Global_Config as gc
import tools
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sqlalchemy import create_engine

import multiprocessing as mp

def f(factor_1, factor_2, shift, field, start_date, end_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    
    sql = """
    select ty.trade_date, ty.stock_code, tx1.factor_value x1, tx2.factor_value x2
    from label.tdailylabel ty
    left join whitelist.tdailywhitelist tw
    on ty.trade_date = tw.trade_date
    and ty.stock_code = tw.stock_code
    left join factor.{factor_1} tx1
    on ty.trade_date = tx1.trade_date
    and ty.stock_code = tx1.stock_code
    left join factor.{factor_2} tx2
    on ty.trade_date = tx2.trade_date
    and ty.stock_code = tx2.stock_code
    where ty.is_trade = 1
    and ty.trade_date >= {start_date}
    and ty.trade_date <= {end_date}
    """
    if field == 'white':
        sql += ' and tw.white = 1 '
    sql = sql.format(factor_1=factor_1, factor_2=factor_2, start_date=start_date, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    x1 = df.loc[:, 'x1'].unstack().shift(shift)
    x2 = df.loc[:, 'x2'].unstack()
    corr = x1.corrwith(x2, axis=1, method='spearman').dropna()
    df = DataFrame({'CORR':corr})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.loc[:, 'SHIFT'] = shift
    df.loc[:, 'FIELD'] = field
    df.loc[:, 'X1'] = factor_1[7:]
    df.loc[:, 'X2'] = factor_2[7:]
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorcorr?charset=utf8")
    df.to_sql('tdailyfactorcorr', engine, schema='factorcorr', index=True, if_exists='append', chunksize=10000, method=tools.mysql_replace_into)

    
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(7)).strftime('%Y%m%d')
    # start_date = '20100101'

    factors = ['value',
               'quality', 'ownership', 'fyqlrl',
               'margin', 
               'moneyflow', 'spread', 
               'momentum', 'sigma', 'skew', 'corrmarket',
               'tr', 'str', 
               'pvcorr',
               'oc', 
               'ao', 'ca',
               'ho', 'lo', 
               'ch', 'cl',  
               'mincorrmarketmean',
               'minmomentummean',
               'minsigmamean',
               'minskewmean',
               'minstrmean',
               ]
    factors = ['tfactor'+i for i in factors]
    
    n = len(factors)
    pool = mp.Pool(4)
    shifts = [1]
    fields = ['white']
    for i in range(n):
        for j in range(n):
            for shift in shifts:
                for field in fields:
                    if i == j:
                        pool.apply_async(func=f, args=(factors[i], factors[j], shift, field, start_date, end_date))
    pool.close()
    pool.join()