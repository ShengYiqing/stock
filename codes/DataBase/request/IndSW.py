# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 23:11:50 2023

@author: admin
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import requests
import tools
import Global_Config as gc
from sqlalchemy import create_engine

url = 'https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls'
headers = {"User-agent": 'chrome'}
r = requests.get(url, headers=headers)
with open ('%s/IndSW/StockClassifyUse_stock.xls'%gc.DATABASE_PATH, 'wb') as f:
    f.write(r.content)

df1 = pd.read_excel('%s/IndSW/SwClassCode_2021.xls'%gc.DATABASE_PATH)
df2 = pd.read_excel('%s/IndSW/StockClassifyUse_stock.xls'%gc.DATABASE_PATH)
df2.sort_values(['股票代码', '计入日期', '更新日期'], inplace=True)
df2 = df2.loc[:, ['股票代码', '行业代码']].groupby('股票代码', as_index=False).last()
df2.loc[:, '股票代码'] = ['0'*(6-len(str(i))) + str(i) for i in df2.loc[:, '股票代码']]
df = pd.merge(left=df2, right=df1, how='left', on='行业代码')
df.columns = ['stock_code', 'ind_code', 'l1_name', 'l2_name', 'l3_name']
df.loc[:, 'REC_CREATOR'] = 'sw'
df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")

df.to_sql('tindsw', engine, schema='indsw', index=False, if_exists='append', method=tools.mysql_replace_into)
