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
    
    f_list = ['fjzc', 'fjyzc', 
              'fyysr', 'fyysrxj', 
              'fml', 'fmlxj', 'fhxlr', 'fzhsy', 'fjyxjll', 
              ]
    
    sql = ' select tmc.stock_code, tmc.factor_value mc, tind.l1_name ind_1, tind.l2_name ind_2, tind.l3_name ind_3 '
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
    left join indsw.tindsw tind
    on tmc.stock_code = tind.stock_code
    """
    trade_cal = tools.get_trade_cal(start_date, end_date)
    
    factor_dic = {}
    # result = {'trade_date':[],
    #           'num_boost_round':[],
    #           'max_bin':[],
    #           'num_leaves':[],
    #           'learning_rate':[],
    #           'm':[],
    #           's':[],
    #           }
    for trade_date in trade_cal:
        # trade_date = '20230130'
        print(trade_date)
        sql_t = sql + ' where tmc.trade_date = %s '%trade_date
        df_sql = pd.read_sql(sql_t, engine).set_index('stock_code')
        df_sql.loc[:, ['fjzc', 'fjyzc', 'fyysr']] = np.exp(df_sql.loc[:, ['fjzc', 'fjyzc', 'fyysr']])
        
        df_sql.loc[:, 'zc'] = df_sql.loc[:, 'fjyzc'] / df_sql.loc[:, 'fjzc']
        
        df_sql.loc[:, 'yy_1'] = df_sql.loc[:, 'fyysr'] / df_sql.loc[:, 'fjzc']
        df_sql.loc[:, 'yy_2'] = df_sql.loc[:, 'fyysrxj'] / df_sql.loc[:, 'fjzc']
        
        df_sql.loc[:, 'yl_1'] = df_sql.loc[:, 'fml'] / df_sql.loc[:, 'fjzc']
        df_sql.loc[:, 'yl_2'] = df_sql.loc[:, 'fmlxj'] / df_sql.loc[:, 'fjzc']
        df_sql.loc[:, 'yl_3'] = df_sql.loc[:, 'fhxlr'] / df_sql.loc[:, 'fjzc']
        df_sql.loc[:, 'yl_4'] = df_sql.loc[:, 'fzhsy'] / df_sql.loc[:, 'fjzc']
        df_sql.loc[:, 'yl_5'] = df_sql.loc[:, 'fjyxjll'] / df_sql.loc[:, 'fjzc']
        
        df_sql.loc[:, ['zc', 'yy_1', 'yy_2', 'yl_1', 'yl_2', 'yl_3', 'yl_4', 'yl_5']] = df_sql.loc[:, ['zc', 'yy_1', 'yy_2', 'yl_1', 'yl_2', 'yl_3', 'yl_4', 'yl_5']].apply(lambda x:x.rank()/x.notna().sum())
        df_sql.loc[:, 'q'] = df_sql.loc[:, ['zc', 'yy_1', 'yy_2', 'yl_1', 'yl_2', 'yl_3', 'yl_4', 'yl_5']].mean(1)
        X = df_sql.loc[:, ['q', 'mc', 'ind_1', 'ind_2', 'ind_3']]
        X.loc[:, 'ind_1'] = X.loc[:, 'ind_1'].astype('category')
        X.loc[:, 'ind_2'] = X.loc[:, 'ind_2'].astype('category')
        X.loc[:, 'ind_3'] = X.loc[:, 'ind_3'].astype('category')
        X.loc[:, 'mc'] = X.loc[:, 'mc'].astype('float32')
        y = df_sql.loc[:, 'q']
        y = tools.standardize(y)
        y_predict = Series(0, index=y.index)
        
        train_data = lgb.Dataset(X, label=y, categorical_feature=['ind_1', 'ind_2', 'ind_3'], free_raw_data=False)
        
    #     n = 256
    #     max_bin_l = [255]
    #     num_leaves_l = [31]
    #     learning_rate_l = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        
    #     for ps in itertools.product(max_bin_l, num_leaves_l, learning_rate_l):
    #         train_data = lgb.Dataset(X, label=y, free_raw_data=False)
    #         num_boost_round = n
    #         max_bin = ps[0]
    #         num_leaves = ps[1]
    #         learning_rate = ps[2]
    #         params = {
    #             'boosting_type': 'gbdt',
    #             'objective': 'regression',
    #             'max_bin': max_bin,
    #             'num_leaves': num_leaves,
    #             'max_depth': -1,
    #             'min_data_in_leaf': 8,
    #             'learning_rate': learning_rate,
    #             'feature_fraction': np.sqrt(0.618),
    #             'feature_fraction_bynode': np.sqrt(0.618),
    #             'bagging_fraction': 0.618,
    #             'bagging_freq': 8,
    #             'verbose': -1,
    #             'force_col_wise': 'true',
    #             'seed': 0,
    #             'metrics': 'L2',
    #             }
    #         cv = lgb.cv(params, train_data, num_boost_round, stratified=False, nfold=10)
        
    #         result['trade_date'].append(trade_date)
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
    # DataFrame(result).set_index(['trade_date', 'learning_rate']).loc[:, 'm'].unstack().to_csv('./e.csv')
        
        num_boost_round = 1024
        params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'max_bin': 255,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 8,
        'learning_rate': 0.05,
        'feature_fraction': np.sqrt(0.618),
        'feature_fraction_bynode': np.sqrt(0.618),
        'bagging_fraction': 0.618,
        'bagging_freq': 8,
        'verbose': -1,
        'force_col_wise': 'true',
        'seed': 0,
        'metrics': 'L2',
        }
        lgbm = lgb.train(params, train_data, num_boost_round)
        y_predict = Series(lgbm.predict(X), index=y.index)
        # y_predict = tools.standardize(y_predict)
        res = y - y_predict
        factor_dic[trade_date] = res
        
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
    df_new.to_sql('tfactorquality', engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)