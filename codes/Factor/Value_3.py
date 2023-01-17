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
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tushare as ts
import itertools
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR

#%%
def generate_factor(start_date, end_date):
    
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    # 净资产
    # 经营资产
    # 营业收入
    # 核心利润
    # 经营现金流
    # 预期收益
    
    f_list = ['fjzc', 'fjyzc', 'fyysr', 'fhxlr', 'fjyxjll', 
              'quality', 'expectedquality',
              ]
    
    sql = ' select tsb.industry industry, tmc.stock_code, tmc.factor_value mc '
    for f in f_list:
        sql = sql + ' , t{f}.factor_value {f} '.format(f=f)
    sql = sql + ' from factor.tfactormc tmc '
    for f in f_list:
        sql = sql + """
        left join factor.tfactor{f} t{f} 
        on tmc.stock_code = t{f}.stock_code 
        and tmc.trade_date = t{f}.trade_date 
        """.format(f=f)
    sql = sql + """
    left join tsdata.ttsstockbasic tsb
    on tmc.stock_code = tsb.stock_code
    """
    trade_cal = tools.get_trade_cal(start_date, end_date)
    
    factor_dic = {}
    for trade_date in trade_cal:
        # trade_date = '20180606'
        print(trade_date)
        sql_t = sql + ' where tmc.trade_date = %s '%trade_date
        df_sql = pd.read_sql(sql_t, engine).set_index('stock_code')
        
        X = df_sql.loc[:, f_list+['industry']]
        X.loc[:, 'industry'] = X.loc[:, 'industry'].astype('category')
        X.loc[:, f_list] = X.loc[:, f_list].astype('float32')
        y = df_sql.loc[:, 'mc']
        y = tools.standardize(y)
        y_predict = Series(0, index=y.index)
        
        train_data = lgb.Dataset(X, label=y, categorical_feature=['industry'], free_raw_data=False)
        
        # result = {'num_boost_round':[],
        #           'max_bin':[],
        #           'num_leaves':[],
        #           'learning_rate':[],
        #           'm':[],
        #           's':[],
        #           }
        
        # max_bin_l = [15]
        # num_leaves_l = [31]
        # learning_rate_l = [0.005, 0.01, 0.02, 0.05, 0.1]
        
        # for ps in itertools.product(max_bin_l, num_leaves_l, learning_rate_l):
        #     train_data = lgb.Dataset(X, label=y, free_raw_data=False)
        #     num_boost_round = 256
        #     max_bin = ps[0]
        #     num_leaves = ps[1]
        #     learning_rate = ps[2]
        #     params = {
        #         'boosting_type': 'gbdt',
        #         'objective': 'regression',
        #         'max_bin': max_bin,
        #         'num_leaves': num_leaves,
        #         'max_depth': -1,
        #         'min_data_in_leaf': 8,
        #         'learning_rate': learning_rate,
        #         'feature_fraction': np.sqrt(0.618),
        #         'feature_fraction_bynode': np.sqrt(0.618),
        #         'bagging_fraction': 0.618,
        #         'bagging_freq': 8,
        #         'verbose': -1,
        #         'force_col_wise': 'true',
        #         'seed': 0,
        #         'metrics': 'L2',
        #         }
        #     cv = lgb.cv(params, train_data, num_boost_round, stratified=False, nfold=10)
            
        #     for n in [256]:
        #         result['num_boost_round'].append(n)
        #         result['max_bin'].append(max_bin)
        #         result['num_leaves'].append(num_leaves)
        #         result['learning_rate'].append(learning_rate)
        #         result['m'].append(cv['l2-mean'][n-1])
        #         result['s'].append(cv['l2-stdv'][n-1])
        #         print('---------------')
        #         print('max_bin=', max_bin)
        #         print('num_leaves=', num_leaves)
        #         print('learning_rate=', learning_rate)
        #         print('num_boost_round=', n)
        #         print('m=', cv['l2-mean'][n-1])
        #         print('s=', cv['l2-stdv'][n-1])
        # DataFrame(result).set_index(['num_boost_round', 'learning_rate']).loc[:, 'm'].unstack().to_csv('./e.csv')
        
        
        
        num_boost_round = 256
        n = 1
        for i in range(n):
            params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'max_bin': 15,
            'num_leaves': 31,
            'max_depth': -1,
            'min_data_in_leaf': 8,
            'learning_rate': 0.02,
            'feature_fraction': np.sqrt(0.618),
            'feature_fraction_bynode': np.sqrt(0.618),
            'bagging_fraction': 0.618,
            'bagging_freq': 8,
            'verbose': -1,
            'force_col_wise': 'true',
            'seed': i,
            }
            lgbm = lgb.train(params, train_data, num_boost_round)
            y_predict = y_predict + Series(lgbm.predict(X.loc[:, f_list+['industry']]), index=y.index)
        y_predict = y_predict / n
        y_predict = tools.standardize(y_predict)
        res = y - y_predict
        factor_dic[trade_date] = res
        
        # model = RandomForestRegressor(n_estimators=1000, min_samples_leaf=8, max_features='sqrt', n_jobs=8, random_state=0)
        # model_fit = model.fit(df_sql.loc[:, f_list], df_sql.loc[:, 'mc'])
        # model_fit.
    factor = DataFrame(factor_dic).T
    
    factor = factor.replace(np.inf, np.nan)
    factor = factor.replace(-np.inf, np.nan)
    factor.index.name = 'trade_date'
    factor.columns.name = 'stock_code'
        
    factor_p = tools.standardize(tools.winsorize(factor))
    df_new = pd.concat([factor, factor_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
    df_new = df_new.stack()
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorvalue', engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(60)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)