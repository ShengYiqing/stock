# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:51:46 2021

@author: admin
"""

import os
import sys
import datetime
import pandas as pd
from pandas import Series, DataFrame

import Global_Config as gc
import tools
import sqlalchemy as sa
from sqlalchemy import create_engine

if __name__ == '__main__':
    
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = '20210104'
    
    sql_trade_cal = """
    select distinct cal_date from ttstradecal where is_open = 1
    """
    
    trade_cal = list(pd.read_sql(sql_trade_cal, engine).loc[:, 'cal_date'])
    trade_cal = list(filter(lambda x:(x>=start_date) & (x<=end_date), trade_cal))
    # trade_cal = ['20220915']
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/mindata?charset=utf8")
    
    sql_upd = """
    ALTER TABLE `mindata`.`mindata%s` 
    RENAME TO  `mindata`.`tmindata%s`

    """
    with engine.connect() as con:
        for trade_date in trade_cal:
            sql_upd_tmp = sql_upd%(trade_date,trade_date)
            con.execute(sql_upd_tmp)