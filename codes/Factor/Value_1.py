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
    
    sql_mc = """
    select tdb.trade_date, tdb.stock_code, tdb.total_mv
    from tsdata.ttsdailybasic tdb
    where tdb.trade_date >= {start_date}
    and tdb.trade_date <= {end_date}
    """.format(start_date=start_date, end_date=end_date)
    mc = pd.read_sql(sql_mc, engine)
    
    sql_ind = """
    select tind.stock_code, tind.l3_name ind
    from indsw.tindsw tind
    """
    ind = pd.read_sql(sql_ind, engine).set_index('stock_code').ind
    
    start_date_fin = tools.trade_date_shift(start_date, 500)
    
    sql_fin = """
    select ann_date, end_date, ann_type, stock_code, financial_index, financial_value 
    from findata.tfindata
    where ann_date >= {start_date}
    and ann_date <= {end_date}
    and substr(end_date, -4) in ('0331', '0630', '0930', '1231')
    and ann_type in ('分析师预期', '定期报告', '业绩预告', '业绩快报')
    and financial_index in ('jzc', 'zzc', 'yysr', 'gmjlr')
    """.format(start_date=start_date_fin, end_date=end_date)
    fin = pd.read_sql(sql_fin, engine).sort_values(['financial_index', 'stock_code', 'end_date', 'ann_date'])
    
    trade_dates = tools.get_trade_cal(start_date, end_date)
    
    factor_dic = {}
    # result = {'trade_date':[],
    #           'num_boost_round':[],
    #           'max_bin':[],
    #           'num_leaves':[],
    #           'learning_rate':[],
    #           'm':[],
    #           's':[],
    #           }
    for trade_date in trade_dates:
        # trade_date = '20230428'
        print(trade_date)
        #y
        y = np.log(mc.loc[mc.trade_date==trade_date].set_index('stock_code').total_mv)
        
        #财务报告
        start_date_tmp = tools.trade_date_shift(trade_date, 500)
        fin_tmp = fin.loc[(fin.end_date>=start_date_tmp)&(fin.end_date<=trade_date)&(fin.ann_type!='分析师预期')]
        #jzc
        jzc = fin_tmp.loc[(fin_tmp.financial_index=='jzc')].groupby('stock_code').last().financial_value
        #zzc
        zzc = fin_tmp.loc[(fin_tmp.financial_index=='zzc')].groupby('stock_code').last().financial_value
        #yysr
        yysr = fin_tmp.loc[(fin_tmp.financial_index=='yysr')].groupby(['stock_code', 'end_date']).last().financial_value
        yysr = yysr.unstack().T
        cols = yysr.columns
        yysr['YYYY'] = [ind[:4] for ind in yysr.index]
        yysr = yysr.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        yysr = yysr.loc[:, cols]
        yysr_ttm = yysr.rolling(4).mean().fillna(method='ffill', limit=1).iloc[-1]
        yysr_ttm.name = 'yysr'
        #gmjlr
        gmjlr = fin_tmp.loc[(fin_tmp.financial_index=='gmjlr')].groupby(['stock_code', 'end_date']).last().financial_value
        gmjlr = gmjlr.unstack().T
        cols = gmjlr.columns
        gmjlr['YYYY'] = [ind[:4] for ind in gmjlr.index]
        gmjlr = gmjlr.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
        gmjlr = gmjlr.loc[:, cols]
        gmjlr_ttm = gmjlr.rolling(4).mean().fillna(method='ffill', limit=1).iloc[-1]
        gmjlr_ttm.name = 'gmjlr'
        
        #预期
        year_1 = trade_date[:4] + '1231'
        year_2 = str(int(trade_date[:4])+1) + '1231'
        fin_tmp = fin.loc[(fin.ann_date>=start_date_tmp)&(fin.ann_date<=trade_date)&(fin.ann_type=='分析师预期')]
        
        con_yysr_1 = fin_tmp.loc[(fin_tmp.financial_index=='yysr') & (fin_tmp.end_date==year_1)]
        con_yysr_1 = con_yysr_1.set_index(['ann_date', 'stock_code']).financial_value.unstack().ewm(halflife=60).mean().iloc[-1]
        
        con_yysr_2 = fin_tmp.loc[(fin_tmp.financial_index=='yysr') & (fin_tmp.end_date==year_2)]
        con_yysr_2 = con_yysr_2.set_index(['ann_date', 'stock_code']).financial_value.unstack().ewm(halflife=60).mean().iloc[-1]
        
        con_gmjlr_1 = fin_tmp.loc[(fin_tmp.financial_index=='gmjlr') & (fin_tmp.end_date==year_1)]
        con_gmjlr_1 = con_gmjlr_1.set_index(['ann_date', 'stock_code']).financial_value.unstack().ewm(halflife=60).mean().iloc[-1]
        
        con_gmjlr_2 = fin_tmp.loc[(fin_tmp.financial_index=='gmjlr') & (fin_tmp.end_date==year_2)]
        con_gmjlr_2 = con_gmjlr_2.set_index(['ann_date', 'stock_code']).financial_value.unstack().ewm(halflife=60).mean().iloc[-1]
        
        coverage = fin_tmp.loc[(fin_tmp.financial_index=='gmjlr') & (fin_tmp.end_date==year_1)].set_index('stock_code').financial_value.groupby('stock_code').count()
        
        df = DataFrame({
            'y': y, 
            'ind': ind, 
            'jzc': jzc, 
            'zzc': zzc, 
            'yysr_ttm': yysr_ttm, 
            'gmjlr_ttm': gmjlr_ttm, 
            'con_yysr_1': con_yysr_1, 
            'con_yysr_2': con_yysr_2, 
            'con_gmjlr_1': con_gmjlr_1, 
            'con_gmjlr_2': con_gmjlr_2, 
            'coverage': coverage, 
            }).dropna(subset='y')
        
        y = df.loc[:, 'y']
        X = df.loc[:, ['ind', 
                       'jzc', 'zzc', 
                       'yysr_ttm', 'gmjlr_ttm', 
                       'con_yysr_1', 'con_yysr_2', 
                       'con_gmjlr_1', 'con_gmjlr_2', 
                       'coverage']]
        X.loc[:, 'ind'] = X.loc[:, 'ind'].astype('category')
        
        y_predict = Series(0, index=y.index)
        
        train_data = lgb.Dataset(X, label=y, categorical_feature=['ind'], free_raw_data=False)
        
    #     n = 255
    #     max_bin_l = [256]
    #     num_leaves_l = [31]
    #     learning_rate_l = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

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
        
        num_boost_round = 255
        params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'max_bin': 256,
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
        res = y_predict - y
        factor_dic[trade_date] = res
        
    factor = DataFrame(factor_dic).T
    
    factor = factor.replace(np.inf, np.nan)
    factor = factor.replace(-np.inf, np.nan)
    factor.index.name = 'trade_date'
    factor.columns.name = 'stock_code'
        
    df_new = DataFrame({'factor_value':factor.stack()})
    df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df_new.to_sql('tfactorvalue', engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

#%%
if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    start_date = '20100101'
    generate_factor(start_date, end_date)