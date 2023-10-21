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
trade_date = '20231019'

with open('D:/stock/Codes/Trade/Results/position/pos.pkl', 'rb') as f:
    position = pickle.load(f)

buy_list = ['601012', '000988', '002027', '600276'
            ]

sell_list= ['002415', '002517', '300033', '688041'
            ]

position.extend(buy_list)
position = list(set(position) - set(sell_list))
# position = ['300308', '002230', '601995', '000988', '600763', 
#             '600570', '603392', '688012', '002517', '603039', 
#             '300059', '002568', '300033', '300418', '603986', 
#             '300496', '688041', '603019']
with open('D:/stock/Codes/Trade//Results/position/pos.pkl', 'wb') as f:
    pickle.dump(position, f)

position.sort()
print('----持股列表----')
print(position)
print('----持股数量----')
print('持股数量: ', len(position))

factors = [
    # 'value', 
    # 'quality', 
    'beta',
    'reversal', 
    'momentum',  
    'seasonality',
    'skew', 
    'cpshl', 
    'crhls', 
    ]

weight_sub = {
    # 'bp':0, 
    # 'quality': 0.01, 
    'beta': 0.01,
    'reversal': -0.01, 
    'momentum': 0.01, 
    'seasonality': 0.01, 
    'skew': 0.01, 
    'cpshl': -0.01, 
    'crhls': 0.01,
    }

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

halflife_mean = 250
halflife_cov = 750

lambda_i = 0.001

end_date = trade_date
start_date = trade_date
trade_dates = tools.get_trade_cal(start_date, end_date)
start_date_ic = tools.trade_date_shift(start_date, 1250)

print('----参数----')
print('halflife_mean: ', halflife_mean)
print('halflife_cov: ', halflife_cov)
print('factors: ', factors)
print('weight_sub: ', weight_sub)
print('lambda_i: ', lambda_i)

#读ic
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")

sql_ic = """
select trade_date, factor_name, 
(ic_d+rank_ic_d)/2 as ic
from tdailyic
where factor_name in {factor_names}
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_ic = sql_ic.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date)
df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name']).loc[:, 'ic'].unstack().loc[:, factors].shift().fillna(method='ffill')

ic_mean = df_ic.ewm(halflife=halflife_mean, min_periods=250).mean().fillna(0)
ic_cov = df_ic.ewm(halflife=halflife_cov, min_periods=60).cov().fillna(0)

weight = DataFrame(0, index=trade_dates, columns=df_ic.columns)
weight.index.name = 'trade_date'
for trade_date in trade_dates:
    mat_ic_cov = ic_cov.loc[trade_date, :]
    mat = mat_ic_cov / np.diag(mat_ic_cov).mean()
    mat = mat + lambda_i * np.diag(np.ones(len(factors)))
    weight.loc[trade_date, :] = (np.linalg.inv(mat).dot(ic_mean.loc[trade_date, :].values) + weight_sub) / 2

sql = tools.generate_sql_y_x(factors + ['mc'], start_date, end_date)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
df.loc[:, 'mc'] = tools.standardize(tools.winsorize(df.loc[:, 'mc']))
for factor in factors:
    df.loc[:, factor] = tools.standardize(tools.winsorize(df.loc[:, factor]))
y = df.loc[:, 'r_d'].unstack()

x = DataFrame(dtype='float64')
for factor in factors:
    df_x = df.loc[:, factor].unstack()
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
print(df.loc[stock_list_old, ['mc'] + factors].mean().round(2))
print('----板块数量----')
print('00', len(list(filter(lambda x:x[0] == '0', position))))
print('30', len(list(filter(lambda x:x[0] == '3', position))))
print('60', len(list(filter(lambda x:x[0:2] == '60', position))))
print('68', len(list(filter(lambda x:x[0:3] == '688', position))))

df.loc[:, ['mc']] = df.loc[:, ['mc']].rank(pct=True)

dp_mask = (2/3<=df.mc)
zp_mask = (1/3<df.mc)&(df.mc<=2/3)
xp_mask = (df.mc<=1/3)

df.loc[dp_mask, 'mc'] = '大盘'
df.loc[zp_mask, 'mc'] = '中盘'
df.loc[xp_mask, 'mc'] = '小盘'

hold_dic = {}
for stock in r_hat_rank.index:
    if stock in stock_ind.index:
        hold_dic[stock] = [stock_ind.loc[stock, 'stock_name'], stock_ind.loc[stock, 'ind_1'], stock_ind.loc[stock, 'ind_2'], stock_ind.loc[stock, 'ind_3'], r_hat_rank.loc[stock], np.around(ret.loc[stock], 3)]
        if stock in df.index:
            hold_dic[stock].extend(list(df.loc[stock].loc[['mc'] + factors]))
        else:
            hold_dic[stock].extend([np.nan] * (1+len(factors)))
    else:
        continue
    if stock_ind.loc[stock, 'ind_1'] in ind_num_dic.keys():
        ind_num_dic[stock_ind.loc[stock, 'ind_1']] += 1
    else:
        ind_num_dic[stock_ind.loc[stock, 'ind_1']] = 1
df_hold = DataFrame(hold_dic).T
df_hold.columns = ['股票名称', '一级行业', '二级行业', '三级行业', '排名', '预期收益', 'mc'] + factors
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
            buy_dic[stock_code].extend(df.loc[stock_code, ['mc'] + factors])

df_buy = DataFrame(buy_dic).T
df_buy.columns = ['股票名称', '一级行业', '二级行业', '三级行业', '排名', '预期收益', 'mc'] + factors
df_buy.index.name = '股票代码'
df_buy.reset_index(inplace=True)
df_buy = df_buy.groupby(['一级行业', '二级行业', '三级行业']).apply(lambda x:x.sort_values('排名', ascending=False, ignore_index=True))
df_buy.index = range(len(df_buy))
df_buy.loc[:, '持仓'] = 0
df_print = pd.concat([df_hold, df_buy])
df_print = df_print.groupby(['一级行业', '二级行业', '三级行业']).apply(lambda x:x.sort_values('排名', ascending=False, ignore_index=True))
df_print.index = range(len(df_print))
df_print = df_print.loc[:, ['股票代码', '股票名称', '一级行业', '二级行业', '三级行业', 'mc', '持仓', '排名', '预期收益']+factors]
df_print.rename({factor:factor + '(%.2f)'%weight.loc[trade_date, factor] for factor in factors}, axis=1, inplace=True)
df_print.to_excel('D:/stock/信号/%s.xlsx'%trade_date)
