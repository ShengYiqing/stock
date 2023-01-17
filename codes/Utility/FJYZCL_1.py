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
    
    shift = 500
    formula = '(t1.money_cap + t1.notes_receiv + t1.accounts_receiv + t1.prepayment + t1.inventories + t1.fix_assets_total + t1.produc_bio_assets + t1.oil_and_gas_assets + t1.intan_assets + t1.receiv_financing + t1.contract_assets) / t1.total_assets'
    table_name_tmp = 'tfactorfjyzcl'
    methods = ['', 'd', 's']
    for method in methods:
        table_name = table_name_tmp + method
        factor = tools.generate_finance_formula(formula, start_date, end_date, shift, method)
        
        factor = factor.replace(np.inf, np.nan)
        factor = factor.replace(-np.inf, np.nan)
        factor.index.name = 'trade_date'
        factor.columns.name = 'stock_code'
        
        factor_p = tools.standardize(tools.winsorize(factor))
        df_new = pd.concat([factor, factor_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
        df_new = df_new.stack()
        df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
        df_new.to_sql(table_name, engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)