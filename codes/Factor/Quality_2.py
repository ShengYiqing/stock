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
    formula = """
    3*tfactorfjyzcl.preprocessed_factor_value
    +2*tfactorfjyzcld.preprocessed_factor_value
    -1*tfactorfjyzcls.preprocessed_factor_value
    -3*tfactorfjrfzl.preprocessed_factor_value
    -2*tfactorfjrfzld.preprocessed_factor_value
    -1*tfactorfjrfzls.preprocessed_factor_value
    -3*tfactorfyszkl.preprocessed_factor_value
    -2*tfactorfyszkld.preprocessed_factor_value
    -1*tfactorfyszkls.preprocessed_factor_value
    +3*tfactorfzzl.preprocessed_factor_value
    +2*tfactorfzzld.preprocessed_factor_value
    -1*tfactorfzzls.preprocessed_factor_value
    +3*tfactorfmll.preprocessed_factor_value
    +2*tfactorfmlld.preprocessed_factor_value
    -1*tfactorfmlls.preprocessed_factor_value
    +3*tfactorfhxlrl.preprocessed_factor_value
    +2*tfactorfhxlrld.preprocessed_factor_value
    -1*tfactorfhxlrls.preprocessed_factor_value
    +3*tfactorfhxfyl.preprocessed_factor_value
    +2*tfactorfhxfyld.preprocessed_factor_value
    -1*tfactorfhxfyls.preprocessed_factor_value
    +3*tfactorfzhsyl.preprocessed_factor_value
    +2*tfactorfzhsyld.preprocessed_factor_value
    -1*tfactorfzhsyls.preprocessed_factor_value
    +3*tfactorfjyxjlll.preprocessed_factor_value
    +2*tfactorfjyxjllld.preprocessed_factor_value
    -1*tfactorfjyxjllls.preprocessed_factor_value"""
    factor = tools.generate_high_order_factor(formula, start_date, end_date)
    
    factor = factor.replace(0, np.nan)
    factor = factor.replace(np.inf, np.nan)
    factor = factor.replace(-np.inf, np.nan)
    factor.index.name = 'trade_date'
    factor.columns.name = 'stock_code'
    
    factor_p = tools.standardize(tools.winsorize(factor))
    df_new = pd.concat([factor, factor_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorquality', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)