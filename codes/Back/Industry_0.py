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
    trade_dates = tools.get_trade_cal(start_date, end_date)
    dic = {
        'ilight': ['纺织', '纺织机械', '轻工机械', '家用电器', '服饰', '家居用品', '陶瓷'], 
        'iheavy': ['运输设备', '航空', '化工机械', '船舶', '工程机械', '农用机械'], 
        'iauto': ['汽车配件', '汽车整车', '摩托车', '汽车服务', ], 
        'isoft': ['软件服务', '互联网', 'IT设备', '通信设备'], 
        'ihard': ['元器件', '半导体'], 
        'ielec': ['电气设备'], 
        'itech': ['机床制造', '专用机械', '电器仪表', '机械基件', ],
        'icons': ['食品', '白酒', '啤酒', '软饮料', '红黄酒', '乳制品', '日用化工', 
                  '旅游景点', '酒店餐饮', '旅游服务', 
                  '影视音像', '文教休闲', '出版业', ], 
        'imed': ['医药商业', '生物制药', '化学制药', '中成药', '医疗保健'], 
        'ibusiness': ['其他商业', '商品城', '商贸代理', '广告包装', '批发业', '仓储物流', 
                      '百货', '超市连锁', '电器连锁', ], 
        }
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    engine_w = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    
    sql = """
    select STOCK_CODE, INDUSTRY from ttsstockbasic
    """
    df_sql = pd.read_sql(sql, engine)
    for ind in dic.keys():
        try:
            sql = """
            CREATE TABLE `factor`.`tfactor%s` (
              `REC_CREATE_TIME` VARCHAR(14) NULL,
              `TRADE_DATE` VARCHAR(8) NOT NULL,
              `STOCK_CODE` VARCHAR(20) NOT NULL,
              `FACTOR_VALUE` DOUBLE NULL,
              `PREPROCESSED_FACTOR_VALUE` DOUBLE NULL,
              PRIMARY KEY (`TRADE_DATE`, `STOCK_CODE`))
            """%ind
            with engine_w.connect() as con:
                con.execute(sql)
        except:
            pass
        ind_sub = dic[ind]
        stocks = df_sql.loc[[i in ind_sub for i in df_sql.INDUSTRY], 'STOCK_CODE']
        df = DataFrame(0, index=trade_dates, columns=df_sql.STOCK_CODE)
        df.index.name = 'trade_date'
        df.columns.name = 'stock_code'
        
        df.loc[:, stocks] = 1
        df_p = tools.standardize(df)
        df_new = pd.concat([df, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
        df_new = df_new.stack()
        df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
        df_new.to_sql('tfactor%s'%ind, engine, schema='factor', if_exists='append', index=True, chunksize=10000, dtype={'STOCK_CODE':VARCHAR(20), 'TRADE_DATE':VARCHAR(8), 'REC_CREATE_TIME':VARCHAR(14)}, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)