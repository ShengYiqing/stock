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
import itertools

import tushare as ts

import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR
import lightgbm as lgb

#%%
def generate_factor(start_date, end_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    
    factors = [
        'quality', 
        'operation', 'gross', 
        'core', 'profitability', 'cash',
        'growth', 'stability', 
        'expectation', 
        'analystcoverage', 
        'conoperation', 'conprofitability', 
        'congrowth', 'conimprovement', 
        'golden', 'goldend',
        ]
    sql = """
    select tdb.stock_code, tdb.trade_date, log(tdb.total_mv) m, 
    log(tdb.total_mv / tdb.pb) b, log(tdb.pb) pb
    tind.l3_name ind
    """
    for factor in factors:
        sql += """
        , t{factor}.factor_value {factor}
        """.format(factor=factor)
    sql += """
    from tsdata.ttsdailybasic tdb
    left join indsw.tindsw tind
    on tdb.stock_code = tind.stock_code
    """
    for factor in factors:
        sql += """
        left join factor.tfactor{factor} t{factor}
        on tdb.stock_code = t{factor}.stock_code
        and tdb.trade_date = t{factor}.trade_date
        """.format(factor=factor)
    sql += """
    where tdb.trade_date >= {start_date} 
    and tdb.trade_date <= {end_date}
    """.format(start_date=start_date, end_date=end_date)

    df = pd.read_sql(sql, engine)
    df.set_index(['trade_date', 'stock_code'], inplace=True)
    df.loc[:, 'ind'] = df.loc[:, 'ind'].astype('category')
    df.loc[:, ['b']+factors] = df.loc[:, ['b']+factors].astype('float32')
    df = df.loc[(df.b.groupby('trade_date').rank(pct=True)>0.025) & (df.m.groupby('trade_date').rank(pct=True)>0.025)]
    trade_dates = tools.get_trade_cal(start_date, end_date)
    
    factor_dic = {}
    for trade_date in trade_dates:
        # trade_date = '20230130'
        print(trade_date)
        df_tmp = df.loc[trade_date]
        
        # X = df_tmp.loc[:, ['ind', 'b', 'e', 's']+factors]
        
        X_1 = df_tmp.loc[:, ['b']]
        y_1 = df_tmp.loc[:, 'm']
        
        train_data = lgb.Dataset(X_1, label=y_1, free_raw_data=False)
        
        num_boost_round = 255
        params = {
        'boosting_type': 'rf', 
        'linear_tree': 'true', 
        'objective': 'regression',
        'max_bin': 255,
        'num_leaves': 255,
        'max_depth': 10,
        'min_data_in_leaf': 63,
        'learning_rate': 0.1,
        # 'feature_fraction': np.sqrt(0.618),
        # 'feature_fraction_bynode': np.sqrt(0.618),
        'bagging_fraction': 0.618,
        'bagging_freq': 8,
        'verbose': -1,
        'force_col_wise': 'true',
        'seed': 0,
        'metrics': 'L2',
        }
        lgbm = lgb.train(params, train_data, num_boost_round)
        y_1_predict = Series(lgbm.predict(X_1), index=y_1.index)
        plt.scatter(X_1.loc[:, 'b'], y_1)
        plt.plot(X_1.loc[:, 'b'].sort_values(), y_1_predict.loc[X_1.loc[:, 'b'].sort_values().index], color='r')
        
        
        y_2 = y_1 - y_1_predict
        
        X_2 = tools.winsorize(df_tmp.loc[:, 'profitability'])
        X_2 = DataFrame({'x':X_2})
        train_data = lgb.Dataset(X_2, label=y_2, free_raw_data=False)
        
        num_boost_round = 63
        params = {
        'boosting_type': 'gbdt', 
        # 'linear_tree': 'true', 
        'objective': 'regression',
        'max_bin': 255,
        'num_leaves': 15,
        'max_depth': 4,
        'min_data_in_leaf': 63,
        'learning_rate': 0.1,
        # 'feature_fraction': np.sqrt(0.618),
        # 'feature_fraction_bynode': np.sqrt(0.618),
        'bagging_fraction': 0.618,
        'bagging_freq': 8,
        'verbose': -1,
        'force_col_wise': 'true',
        'seed': 0,
        'metrics': 'L2',
        }
        lgbm = lgb.train(params, train_data, num_boost_round)
        y_2_predict = Series(lgbm.predict(X_2), index=y_2.index)
        plt.scatter(X_2.loc[:, 'x'], y_2)
        plt.plot(X_2.loc[:, 'x'].sort_values(), y_2_predict.loc[X_2.loc[:, 'x'].sort_values().index], color='r')
        
        y_3 = y_2 - y_2_predict
        
        X = df_tmp.loc[:, ['ind', 's']]
        # print(dict(zip(lgbm.feature_name(), lgbm.feature_importance())))
        # plt.scatter(y, res)
        # print(y.corr(res))
        # print(1 - (res**2).sum() / (y**2).sum())
        factor_dic[trade_date] = res
        
    factor = DataFrame(factor_dic).T
    factor.index.name = 'trade_date'
    factor.columns.name = 'stock_code'
    factor = tools.neutralize(factor)
    df = DataFrame({'factor_value':factor.stack()})
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df.to_sql('tfactorvalue', engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    end_date = '20230509'
    start_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')
    start_date = '20210101'
    generate_factor(start_date, end_date)
