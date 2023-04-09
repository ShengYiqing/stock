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
from sklearn.linear_model import LinearRegression

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
trade_date = '20230406'

with open('D:/stock/Codes/Trade/Results/position/pos.pkl', 'rb') as f:
    position = pickle.load(f)

buy_list = ['600590', '301103', '605555', '300841', '605507', 
            
            ]

sell_list= ['002001', '000333', '600566', '603678', '300206', 
            '603355'
            ]

position.extend(buy_list)
position = list(set(position) - set(sell_list))
with open('D:/stock/Codes/Trade//Results/position/pos.pkl', 'wb') as f:
    pickle.dump(position, f)

position.sort()
print('----持股列表----')
print(position)
print('----持股数量----')
print('持股数量: ', len(position))

halflife_mean = 250
halflife_cov = 750
lambda_i = 0.01

factors = [
    'operation', 'profitability', 'growth', 
    'momentum', 'volatility', 'liquidity', 'corrmarket',
    'dailytech', 'hftech', 
    ]
neutral_list = ['operation', 'profitability', 'growth', ]

factor_value_type_dic = {factor: 'neutral_factor_value' if factor in neutral_list else 'preprocessed_factor_value' for factor in factors}

ic_sub = {'mc':0.01, 'bp':0.01}
ic_sub = {}

for factor in factors:
    if factor not in ic_sub.keys():
        ic_sub[factor] = 0
ic_sub = Series(ic_sub)

end_date = trade_date
start_date = trade_date
trade_dates = tools.get_trade_cal(start_date, end_date)
start_date_ic = tools.trade_date_shift(start_date, 1250)

print('----参数----')
print('halflife_mean: ', halflife_mean)
print('halflife_cov: ', halflife_cov)
print('factors: ', factors)
print('ic_sub: ', ic_sub)
print('lambda_i: ', lambda_i)

#读ic
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")

sql_ic = """
select trade_date, factor_name, 
(ic_m+rank_ic_m)/2 as ic_m, 
(ic_w+rank_ic_w)/2 as ic_w, 
(ic_d+rank_ic_d)/2 as ic_d
from tdailyic
where factor_name in {factor_names}
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_ic = sql_ic.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date)
df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name'])
df_ic_m = df_ic.loc[:, 'ic_m'].unstack().loc[:, factors].shift(21).fillna(method='ffill')
df_ic_w = df_ic.loc[:, 'ic_w'].unstack().loc[:, factors].shift(6).fillna(method='ffill')
df_ic_d = df_ic.loc[:, 'ic_d'].unstack().loc[:, factors].shift(2).fillna(method='ffill')

ic_dic = {'d':df_ic_d, 'w':df_ic_w, 'm':df_ic_m}

sql_h = """
select trade_date, factor_name, 
(h_m+rank_h_m)/2 as h_m, 
(h_w+rank_h_w)/2 as h_w, 
(h_d+rank_h_d)/2 as h_d
from tdailyh
where factor_name in {factor_names}
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_h = sql_h.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date)
df_h = pd.read_sql(sql_h, engine).set_index(['trade_date', 'factor_name'])
df_h_m = df_h.loc[:, 'h_m'].unstack().loc[:, factors].shift(21).fillna(method='ffill')
df_h_w = df_h.loc[:, 'h_w'].unstack().loc[:, factors].shift(6).fillna(method='ffill')
df_h_d = df_h.loc[:, 'h_d'].unstack().loc[:, factors].shift(2).fillna(method='ffill')

h_dic = {'d':df_h_d, 'w':df_h_w, 'm':df_h_m}

sql_tr = """
select trade_date, factor_name, 
(tr_m+rank_tr_m)/2 as tr_m, 
(tr_w+rank_tr_w)/2 as tr_w, 
(tr_d+rank_tr_d)/2 as tr_d
from tdailytr
where factor_name in {factor_names}
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_tr = sql_tr.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date)
df_tr = pd.read_sql(sql_tr, engine).set_index(['trade_date', 'factor_name'])
df_tr_m = df_tr.loc[:, 'tr_m'].unstack().loc[:, factors].fillna(method='ffill')
df_tr_w = df_tr.loc[:, 'tr_w'].unstack().loc[:, factors].fillna(method='ffill')
df_tr_d = df_tr.loc[:, 'tr_d'].unstack().loc[:, factors].fillna(method='ffill')

tr_dic = {'d':df_tr_d, 'w':df_tr_w, 'm':df_tr_m}

weight_dic = {}
for t in ic_dic.keys():
    df_ic = ic_dic[t]
    df_h = h_dic[t]
    df_tr = tr_dic[t]
    
    ic_mean = df_ic.ewm(halflife=halflife_mean, min_periods=250).mean().fillna(0)
    
    ic_std = df_ic.ewm(halflife=halflife_cov, min_periods=250).std().fillna(0)
    ic_corr = df_ic.ewm(halflife=halflife_cov, min_periods=250).corr().fillna(0)
    
    h_mean = df_h.ewm(halflife=halflife_mean, min_periods=250).mean().fillna(0)
    
    tr_mean = df_tr.ewm(halflife=halflife_mean, min_periods=250).mean().fillna(0)
    
    weight = DataFrame(0, index=trade_dates, columns=df_ic.columns)
    weight.index.name = 'trade_date'
    for trade_date in trade_dates:
        mat_ic_corr = ic_corr.loc[trade_date, :]
        mat_ic_corr_tune = mat_ic_corr ** 3
        
        ic_s = ic_std.loc[trade_date, :]
        
        h = h_mean.loc[trade_date, :] ** 0.25
        tr = tr_mean.loc[trade_date, :]
        mat_ic_s_tune = np.diag(ic_s * h)
        
        mat_ic_cov = mat_ic_s_tune.dot(mat_ic_corr_tune).dot(mat_ic_s_tune)
        mat = mat_ic_cov / np.diag(mat_ic_cov).mean()
        mat = mat + lambda_i * np.diag(np.ones(len(factors)))
        weight.loc[trade_date, :] = np.linalg.inv(mat).dot((ic_mean.loc[trade_date, :] * tr).values)
    
    weight_dic[t] = weight.div(weight.std(1), axis=0)
weight_dic['d'] = 1 * weight_dic['d']
weight_dic['w'] = 1 * weight_dic['w']
weight_dic['m'] = 1 * weight_dic['m']
weight = pd.concat([weight.stack() for weight in weight_dic.values()], axis=1).mean(1).unstack()

start_date = trade_date
sql = tools.generate_sql_y_x(factors, start_date, end_date, factor_value_type_dic=factor_value_type_dic)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])

r_hat = DataFrame(dtype='float64')
for factor in factors:
    r_hat = r_hat.add((df.loc[:, factor].unstack().mul(weight.loc[:, factor], axis=0)), fill_value=0)

# x = r_hat

# sql = """
# select tlabel.trade_date trade_date, tlabel.stock_code stock_code, tind.ind_code ind, tmc.preprocessed_factor_value mc, tbp.preprocessed_factor_value bp 
# from label.tdailylabel tlabel
# left join indsw.tindsw tind
# on tlabel.stock_code = tind.stock_code
# left join factor.tfactormc tmc
# on tlabel.stock_code = tmc.stock_code
# and tlabel.trade_date = tmc.trade_date
# left join factor.tfactorbp tbp
# on tlabel.stock_code = tbp.stock_code
# and tlabel.trade_date = tbp.trade_date
# where tlabel.trade_date = {trade_date}
# and tlabel.stock_code in {stock_codes}""".format(trade_date=trade_date, stock_codes=tuple(x.columns))
# engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
# df_n = pd.read_sql(sql, engine)
# df_n = df_n.set_index(['trade_date', 'stock_code'])
# x = x.stack()
# x.name = 'x'
# data = pd.concat([x, df_n], axis=1).dropna()

# def f(data):
#     X = pd.concat([pd.get_dummies(data.ind)], axis=1).fillna(0)
#     y = data.loc[:, 'x']
#     model = LinearRegression(n_jobs=-1)
#     model.fit(X, y)
#     y_predict = Series(model.predict(X), index=y.index)
    
#     res = y - y_predict
#     return res
# x_n = data.groupby('trade_date').apply(f).unstack()
# x_n = x_n.reset_index(0, drop=True).unstack().T
# r_hat = x_n

stocks_all = sorted(list(set(list(r_hat.columns)+(position))))
r_hat = DataFrame(r_hat, columns=stocks_all)
ret = r_hat.loc[trade_date, :].loc[position].sort_values(ascending=False)
r_hat_rank = r_hat.loc[trade_date, :].rank().loc[position].sort_values(ascending=False)

hold_dic = {}
for stock in r_hat_rank.index:
    if stock in stock_ind.index:
        hold_dic[stock] = [stock_ind.loc[stock, 'stock_name'], stock_ind.loc[stock, 'ind_1'], stock_ind.loc[stock, 'ind_2'], stock_ind.loc[stock, 'ind_3'], r_hat_rank.loc[stock], np.around(ret.loc[stock], 3)]
        if stock in df.loc[trade_date, :].index:
            hold_dic[stock].extend(list(df.loc[trade_date, :].loc[stock].loc[factors]))
        else:
            hold_dic[stock].extend([np.nan] * len(factors))
    else:
        continue
    if stock_ind.loc[stock, 'ind_1'] in ind_num_dic.keys():
        ind_num_dic[stock_ind.loc[stock, 'ind_1']] += 1
    else:
        ind_num_dic[stock_ind.loc[stock, 'ind_1']] = 1
df_hold = DataFrame(hold_dic).T
df_hold.columns = ['股票名称', '一级行业', '二级行业', '三级行业', '排名', '预期收益'] + factors
df_hold.index.name = '股票代码'
df_hold.reset_index(inplace=True)
df_hold = df_hold.groupby(['一级行业', '二级行业', '三级行业']).apply(lambda x:x.sort_values('排名', ascending=False, ignore_index=True))
df_hold.index = range(len(df_hold))
df_hold.loc[:, '持仓'] = 1
# print(df_hold)

df = df.loc[trade_date, :]
stock_list_old = list(set(r_hat_rank.index).intersection(set(df.index)))
print('----持股暴露----')
print(df.loc[stock_list_old, factors].mean().round(2))
print('----板块数量----')
print('00', len(list(filter(lambda x:x[0] == '0', position))))
print('30', len(list(filter(lambda x:x[0] == '3', position))))
print('60', len(list(filter(lambda x:x[0:2] == '60', position))))
print('68', len(list(filter(lambda x:x[0:3] == '688', position))))
print('----行业数量----')
print(ind_num_dic)
print('---%s---'%trade_date)

ret = r_hat.loc[trade_date, :].sort_values(ascending=False)
r_hat_rank = r_hat.loc[trade_date, :].rank().sort_values(ascending=False)
n = 5

buy_dic = {}
# ind_num_dic = {i : 0 for i in ind.loc[:, 'ind_3'] if len(set(list(ind.loc[ind.loc[:, 'ind_1']==i, 'ind_3'])) & set(gc.WHITE_INDUSTRY_LIST)) > 0}

for ind in gc.WHITE_INDUSTRY_LIST:
    stocks = list(stock_ind.index[stock_ind.loc[:, 'ind_3']==ind])
    stocks = list(set(stocks).intersection(stocks_all) - set(position))
    ret_tmp = ret.loc[stocks].sort_values(ascending=False)
    r_hat_rank_tmp = r_hat_rank.loc[stocks].sort_values(ascending=False)
    m = min(n, len(stocks))
    for i in range(m):
        stock_code = r_hat_rank_tmp.index[i]
        if stock_code not in position:
            buy_dic[stock_code] = [stock_ind.loc[stock_code, 'stock_name'], 
                                   stock_ind.loc[stock_code, 'ind_1'], 
                                   stock_ind.loc[stock_code, 'ind_2'], 
                                   stock_ind.loc[stock_code, 'ind_3'], 
                                   r_hat_rank_tmp.loc[stock_code], 
                                   np.around(ret_tmp.loc[stock_code], 3),]
            buy_dic[stock_code].extend(df.loc[stock_code, factors])

df_buy = DataFrame(buy_dic).T
df_buy.columns = ['股票名称', '一级行业', '二级行业', '三级行业', '排名', '预期收益'] + factors
df_buy.index.name = '股票代码'
df_buy.reset_index(inplace=True)
df_buy = df_buy.groupby(['一级行业', '二级行业', '三级行业']).apply(lambda x:x.sort_values('排名', ascending=False, ignore_index=True))
df_buy.index = range(len(df_buy))
df_buy.loc[:, '持仓'] = 0
df_print = pd.concat([df_hold, df_buy])
df_print = df_print.groupby(['一级行业', '二级行业', '三级行业']).apply(lambda x:x.sort_values('排名', ascending=False, ignore_index=True))
df_print.index = range(len(df_print))
df_print = df_print.loc[:, ['股票代码', '股票名称', '一级行业', '二级行业', '三级行业', '持仓', '排名', '预期收益']+factors]
df_print.to_excel('D:/stock/信号/%s.xlsx'%trade_date)
