# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:53:44 2021

@author: admin
"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import pickle
import time
import datetime
import Config
import sys
from sqlalchemy import create_engine

sys.path.append(Config.GLOBALCONFIG_PATH)

import tools
import Global_Config as gc

if __name__ == '__main__':
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    sql_stock_ind = """
    select t1.stock_code, t1.name stock_name, t2.l1_name ind_1, t2.l2_name ind_2, t2.l3_name ind_3
    from tsdata.ttsstockbasic t1 
    left join indsw.tindsw t2
    on t1.stock_code = t2.stock_code
    where not isnull(t2.l1_name)"""
    stock_ind = pd.read_sql(sql_stock_ind, engine).set_index('stock_code')
    sql_ind = """
    select l1_name ind_1, l2_name ind_2, l3_name ind_3 from indsw.tindsw 
    group by l1_name, l2_name, l3_name
    order by l1_name, l2_name, l3_name"""
    ind = pd.read_sql(sql_ind, engine)
    ind_num_dic = {i : 0 for i in ind.loc[:, 'ind_1'] if len(set(list(ind.loc[ind.loc[:, 'ind_1']==i, 'ind_3'])) & set(gc.WHITE_INDUSTRY_LIST)) > 0}
    
    trade_date = datetime.datetime.today().strftime('%Y%m%d')
    trade_date = '20230228'
    
    with open('./Results/position/pos.pkl', 'rb') as f:
        position = pickle.load(f)
        
    buy_list = ['002430', '601231', '600089', '600967', '600271', '603444']

    sell_list= ['600718', '688559', '300124', '002507', '688396', '300457']
    
    position.extend(buy_list)
    position = list(set(position) - set(sell_list))
    with open('./Results/position/pos.pkl', 'wb') as f:
        pickle.dump(position, f)
        
    position.sort()
    print(position)
    print(len(position))
    
    white_threshold = 0.8
    is_neutral = 0
    factor_value_type = 'neutral_factor_value' if is_neutral else 'preprocessed_factor_value'
    halflife_ic_mean = 250
    halflife_ic_cov = 750
    lambda_ic = 50
    lambda_i = 50
    
    factors = [
        'value', 'quality', 
        'dailytech', 'hftech', 
        ]

    ic_sub = {'mc':0.01, 'bp':0.01}
    ic_sub = {}
    
    for factor in factors:
        if factor not in ic_sub.keys():
            ic_sub[factor] = 0
    ic_sub = Series(ic_sub)
    
    end_date = trade_date
    start_date = tools.trade_date_shift(trade_date, 1000)
    trade_dates = tools.get_trade_cal(start_date, end_date)
    
    print('halflife_ic_mean: ', halflife_ic_mean)
    print('halflife_ic_cov: ', halflife_ic_cov)
    print('lambda_ic: ', lambda_ic)
    print('lambda_i: ', lambda_i)
    print('factors: ', factors)
    print('ic_sub: ', ic_sub)
    
    #读ic
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/ic?charset=utf8")
    
    sql_ic = """
    select trade_date, factor_name, 
    (ic_m+rank_ic_m)/2 as ic_m, 
    (ic_w+rank_ic_w)/2 as ic_w, 
    (ic_d+rank_ic_d)/2 as ic_d
    from tdailyic
    where factor_name in {factor_names}
    and trade_date >= {start_date}
    and trade_date <= {end_date}
    and white_threshold = {white_threshold}
    and is_neutral = {is_neutral}
    """
    sql_ic = sql_ic.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date, end_date=end_date, white_threshold=white_threshold, is_neutral=is_neutral)
    df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name'])
    df_ic_m = df_ic.loc[:, 'ic_m'].unstack().loc[:, factors].shift(21).fillna(method='ffill')
    df_ic_w = df_ic.loc[:, 'ic_w'].unstack().loc[:, factors].shift(6).fillna(method='ffill')
    df_ic_d = df_ic.loc[:, 'ic_d'].unstack().loc[:, factors].shift(2).fillna(method='ffill')
    df_ic = df_ic_d

    ic_mean = df_ic.ewm(halflife=halflife_ic_mean).mean()
    ic_cov = df_ic.ewm(halflife=halflife_ic_cov).cov().fillna(0)

    weight = DataFrame(0, index=trade_dates, columns=df_ic_m.columns)
    for trade_date in trade_dates:
        mat_ic = ic_cov.loc[trade_date, :].values
        mat_ic = mat_ic / np.trace(mat_ic)
        mat_i = np.diag(np.ones(len(factors)))
        mat_i = mat_i / np.trace(mat_i)
        mat = lambda_ic * mat_ic + lambda_i * mat_i
        weight.loc[trade_date, :] = np.linalg.inv(mat).dot(ic_mean.loc[trade_date, :].values / 2)
        
    start_date = trade_date
    sql = tools.generate_sql_y_x(factors, start_date, end_date, white_threshold=white_threshold, factor_value_type=factor_value_type)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    
    r_hat = DataFrame(dtype='float64')
    for factor in factors:
        print(factor)
        r_hat = r_hat.add((df.loc[:, factor].unstack().mul(weight.loc[:, factor], axis=0)), fill_value=0)

    stocks_all = sorted(list(set(list(r_hat.columns)+(position))))
    r_hat = DataFrame(r_hat, columns=stocks_all)
    ret = r_hat.loc[trade_date, :].loc[position].sort_values(ascending=False)
    r_hat_rank = r_hat.loc[trade_date, :].rank().loc[position].sort_values(ascending=False)
    
    for stock in r_hat_rank.index:
        s = """{:>8} {:>8} {:>8} {:>8} {:>8}"""
        if stock[:6] in stock_ind.index:
            print(s.format(stock[:6], stock_ind.loc[stock[:6], 'stock_name'], r_hat_rank.loc[stock], np.around(ret.loc[stock], 3), stock_ind.loc[stock[:6], 'ind_1']))
        else:
            print(s.format(stock[:6], ' ', r_hat_rank.loc[stock], np.around(ret.loc[stock], 3), ' '))    
        if stock_ind.loc[stock[:6], 'ind_1'] in ind_num_dic.keys():
            ind_num_dic[stock_ind.loc[stock[:6], 'ind_1']] += 1
        else:
            ind_num_dic[stock_ind.loc[stock[:6], 'ind_1']] = 1
    df = df.loc[trade_date, :]
    stock_list_old = list(set(r_hat_rank.index).intersection(set(df.index)))
    print(df.loc[stock_list_old, factors].mean().round(2))
    print('00', len(list(filter(lambda x:x[0] == '0', position))))
    print('30', len(list(filter(lambda x:x[0] == '3', position))))
    print('60', len(list(filter(lambda x:x[0:2] == '60', position))))
    print('68', len(list(filter(lambda x:x[0:3] == '688', position))))
    print(ind_num_dic)
    print('---%s---'%trade_date)
    
    ret = r_hat.loc[trade_date, :].sort_values(ascending=False)
    r_hat_rank = r_hat.loc[trade_date, :].rank().sort_values(ascending=False)
    n = 10
    
    for ind in ind_num_dic.keys():
        stocks = list(stock_ind.index[stock_ind.loc[:, 'ind_1']==ind])
        stocks = list(set(stocks).intersection(stocks_all) - set(position))
        ret_tmp = ret.loc[stocks].sort_values(ascending=False)
        r_hat_rank_tmp = r_hat_rank.loc[stocks].sort_values(ascending=False)
        m = min(n, len(stocks))
        for i in range(m):
            stock_code = r_hat_rank_tmp.index[i]
            if stock_code not in position:
                print(s.format(stock_code, 
                                stock_ind.loc[stock_code, 'stock_name'], 
                                r_hat_rank_tmp.loc[stock_code], 
                                np.around(ret_tmp.loc[stock_code], 3), 
                                ind))
                