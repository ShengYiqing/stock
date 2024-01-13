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

trade_date = datetime.datetime.today().strftime('%Y%m%d')
trade_date = '20240111'

with open('D:/stock/Codes/Trade/Results/position/pos.pkl', 'rb') as f:
    position = pickle.load(f)

buy_list = ['600519', '601985', '002230', '601699', 
            ]

sell_list= ['000858', '000568', '600887', '603345', 
            '603605', '002594', '002714', '603225', 
            ]

position.extend(buy_list)
position = list(set(position) - set(sell_list))
# position = ['688036', '002568', '002027', '300866', '000858', 
#             '600398', '600809', '603605', 
#             ]
with open('D:/stock/Codes/Trade//Results/position/pos.pkl', 'wb') as f:
    pickle.dump(position, f)

position.sort()
print('----持股列表----')
print(position)
print('----持股数量----')
print('持股数量: ', len(position))

halflife = 250

lambda_i = 0.001
print('halflife', halflife)

factors = [
    'beta', 
    'mc', 
    'bp',
    # 'jump', 
    'reversal', 
    'momentum',  
    'seasonality',
    'skew',
    'crhl', 
    'cphl', 
    ]

weight_sub = {
    'beta': 0.01, 
    'mc': 0.01, 
    'bp': 0.01,
    'beta': 0.01,
    # 'jump': 0.005, 
    'reversal': 0.01, 
    'momentum': 0.01,  
    'seasonality': 0.01,
    'skew': 0.01,
    'crhl': 0.01, 
    'cphl': 0.01, 
    }

a = 0.5

for factor in factors:
    if factor not in weight_sub.keys():
        weight_sub[factor] = 0
weight_sub = Series(weight_sub)

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
ind_num_dic = {i : 0 for i in ind.loc[:, 'ind_1']}
ind_num_dic_3 = {i : 0 for i in ind.loc[:, 'ind_3']}

end_date = trade_date
start_date = trade_date
trade_dates = tools.get_trade_cal(start_date, end_date)
start_date_ic = tools.trade_date_shift(start_date, 1250)

print('----参数----')
print('halflife: ', halflife)
print('factors: ', factors)
print('weight_sub: ', weight_sub)
print('lambda_i: ', lambda_i)

#读ic
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")

sql_ic = """
select trade_date, factor_name, 
ic ic
from tdailyfactorevaluation
where factor_name in {factor_names}
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_ic = sql_ic.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date)
df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name']).loc[:, 'ic'].unstack().loc[:, factors].shift().fillna(method='ffill')

ic_mean = df_ic.ewm(halflife=halflife, min_periods=250).mean().fillna(0)
ic_cov = df_ic.ewm(halflife=halflife, min_periods=60).cov().fillna(0)

weight = DataFrame(0, index=trade_dates, columns=df_ic.columns)
weight.index.name = 'trade_date'
for trade_date in trade_dates:
    mat_ic_cov = ic_cov.loc[trade_date, :]
    mat = mat_ic_cov / np.diag(mat_ic_cov).mean()
    mat = mat + lambda_i * np.diag(np.ones(len(factors)))
    weight.loc[trade_date, :] = a * weight_sub + (1 - a) * np.linalg.inv(mat).dot(ic_mean.loc[trade_date, :].values)

sql = tools.generate_sql_y_x(factors, start_date, end_date)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
for factor in factors:
    df.loc[:, factor] = tools.standardize(df.loc[:, factor].rank(pct=True))
    
y = df.loc[:, 'r'].unstack()

x = DataFrame(dtype='float64')
for factor in factors:
    df_x = df.loc[:, factor].unstack()
    df_x = df_x.rank(axis=1, pct=True)
    df_x = tools.standardize(df_x)
    x = x.add(df_x.mul(weight.loc[:, factor], axis=0), fill_value=0)
# x = tools.neutralize(x, ['mc', 'bp', 'sigma', 'tr']).reset_index(-1, drop=True).unstack()
r_hat = x
stocks_all = sorted(list(set(list(r_hat.columns)+(position))))
r_hat = DataFrame(r_hat, columns=stocks_all)
ret = r_hat.loc[trade_date, :].loc[position].sort_values(ascending=False)
r_hat_rank = r_hat.loc[trade_date, :].rank().loc[position].sort_values(ascending=False)

df = df.loc[trade_date, :]
stock_list_old = list(set(r_hat_rank.index).intersection(set(df.index)))
print('----持股暴露----')
print(df.loc[stock_list_old, factors].mean().round(2))
print('----板块数量----')
print('00', len(list(filter(lambda x:x[0] == '0', position))))
print('30', len(list(filter(lambda x:x[0] == '3', position))))
print('60', len(list(filter(lambda x:x[0:2] == '60', position))))
print('68', len(list(filter(lambda x:x[0:3] == '688', position))))

hold_dic = {}
for stock in r_hat_rank.index:
    if stock in stock_ind.index:
        hold_dic[stock] = [stock_ind.loc[stock, 'stock_name'], stock_ind.loc[stock, 'ind_1'], stock_ind.loc[stock, 'ind_2'], stock_ind.loc[stock, 'ind_3'], r_hat_rank.loc[stock], np.around(ret.loc[stock], 3)]
        if stock in df.index:
            hold_dic[stock].extend(list(df.loc[stock].loc[factors]))
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

print('----行业数量----')
print(ind_num_dic)
print('---%s---'%trade_date)

ret = r_hat.loc[trade_date, :].sort_values(ascending=False)
r_hat_rank = r_hat.loc[trade_date, :].rank().sort_values(ascending=False)
n = 1000

buy_dic = {}
# ind_num_dic = {i : 0 for i in ind.loc[:, 'ind_3'] if len(set(list(ind.loc[ind.loc[:, 'ind_1']==i, 'ind_3'])) & set(gc.WHITE_INDUSTRY_LIST)) > 0}

for ind in ind_num_dic_3.keys():
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
df_print.rename({factor:factor + '(%.2f)'%weight.loc[trade_date, factor] for factor in factors}, axis=1, inplace=True)
df_print.to_excel('D:/stock/信号/%s.xlsx'%trade_date)
