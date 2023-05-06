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
    
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    files = os.listdir('D:/stock/DataBase/Data/factor')
    tables = [i.split('.')[0] for i in files]
    for table in tables:
        try:
            sql = """
            ALTER TABLE `factor`.`%s` 
            DROP COLUMN `NEUTRAL_FACTOR_VALUE`,
            DROP COLUMN `PREPROCESSED_FACTOR_VALUE`,
            CHANGE COLUMN `STOCK_CODE` `STOCK_CODE` VARCHAR(18) NOT NULL
            """%table
            with engine.connect() as con:
                con.execute(sql)
        except:
            pass