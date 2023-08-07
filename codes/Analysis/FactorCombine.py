
import datetime
import os
import sys
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import statsmodels.api as sm
halflife_mean = 250
halflife_cov = 750

seasonal_n_mean = 20
n_1 = 250 - seasonal_n_mean
n_2 = n_1 + 250
n_3 = n_2 + 250
n_4 = n_3 + 250
n_5 = n_4 + 250
weight_s = 0.382


lambda_i = 0.001
print('halflife_mean', halflife_mean)
print('halflife_cov', halflife_cov)
print('seasonal_n_mean', seasonal_n_mean)


factors = [
    'beta', 
    'reversal', 
    # 'tr', 
    # 'mc', 
    # 'bp', 
    # 'quality', 
    # 'expectation', 
    'cxx', 
    'hftech', 
    ]

weight_sub = {
    'beta':0.01, 
    'reversal': -0.01, 
    # 'tr': -0.01, 
    # 'mc': -0.01, 
    # 'bp': 0.01, 
    # 'quality':0.01, 
    # 'expectation': 0.02, 
    'cxx': 0.01, 
    'hftech': 0.01
    }
# weight_sub = {}
for factor in factors:
    if factor not in weight_sub.keys():
        weight_sub[factor] = 0
weight_sub = Series(weight_sub)

start_date = '20180101'
if datetime.datetime.today().strftime('%H%M') < '2200':
    end_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')
else:
    end_date = datetime.datetime.today().strftime('%Y%m%d')

# end_date = '20230302'

print('factors: ', factors)
print('weight_sub: ', weight_sub)

trade_dates = tools.get_trade_cal(start_date, end_date)
start_date_ic = tools.trade_date_shift(start_date, 1500)

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
df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name']).loc[:, 'ic'].unstack().loc[:, factors].shift(1).fillna(method='ffill')

sql_h = """
select trade_date, factor_name, 
(h_d+rank_h_d)/2 as h
from tdailyh
where factor_name in {factor_names}
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_h = sql_h.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date)
df_h = pd.read_sql(sql_h, engine).set_index(['trade_date', 'factor_name']).loc[:, 'h'].unstack().loc[:, factors].shift(1).fillna(method='ffill')

sql_tr = """
select trade_date, factor_name, 
(tr_d+rank_tr_d)/2 as tr
from tdailytr
where factor_name in {factor_names}
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_tr = sql_tr.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date)
df_tr = pd.read_sql(sql_tr, engine).set_index(['trade_date', 'factor_name']).loc[:, 'tr'].unstack().loc[:, factors].fillna(method='ffill')


ic_mean = df_ic.ewm(halflife=halflife_mean, min_periods=60).mean().fillna(0)

# ic_mean_s = df_ic.rolling(seasonal_n_mean, min_periods=5).mean().fillna(0)
# ic_mean_s_1 = ic_mean_s.shift(n_1)
# ic_mean_s_2 = ic_mean_s.shift(n_2)
# ic_mean_s_3 = ic_mean_s.shift(n_3)
# ic_mean_s_4 = ic_mean_s.shift(n_4)
# ic_mean_s_5 = ic_mean_s.shift(n_5)
# ic_mean_s = pd.concat([ic_mean_s_1, ic_mean_s_2, ic_mean_s_3, ic_mean_s_4, ic_mean_s_5], axis=1, keys=[1, 2, 3, 4, 5]).stack().mean(1).unstack()

# ic_mean.loc[:, ['quality', 'expectation']] = (1 - weight_s) * ic_mean + weight_s * ic_mean_s
# ic_mean.fillna(0, inplace=True)

ic_std = df_ic.ewm(halflife=halflife_cov, min_periods=60).std().fillna(0)

ic_corr = df_ic.ewm(halflife=halflife_cov, min_periods=60).corr().fillna(0)
ic_cov = df_ic.ewm(halflife=halflife_cov, min_periods=60).cov().fillna(0)
h_mean = df_h.ewm(halflife=halflife_mean, min_periods=60).mean().fillna(0)

tr_mean = df_tr.ewm(halflife=halflife_mean, min_periods=60).mean().fillna(0)

weight = DataFrame(0, index=trade_dates, columns=df_ic.columns)
weight.index.name = 'trade_date'
for trade_date in trade_dates:
    mat_ic_corr = ic_corr.loc[trade_date, :]
    mat_ic_corr_tune = mat_ic_corr ** 3
    
    ic_s = ic_std.loc[trade_date, :]
    
    h = h_mean.loc[trade_date, :] ** (1/4)
    tr = np.exp(tr_mean.loc[trade_date, :])
    mat_ic_s_tune = np.diag(ic_s * h)
    
    mat_ic_cov = mat_ic_s_tune.dot(mat_ic_corr_tune).dot(mat_ic_s_tune)
    mat = mat_ic_cov / np.diag(mat_ic_cov).mean()
    mat = mat + lambda_i * np.diag(np.ones(len(factors)))
    weight.loc[trade_date, :] = (np.linalg.inv(mat).dot((ic_mean.loc[trade_date, :] * tr).values) + weight_sub) / 2
# for trade_date in trade_dates:
#     mat_ic_cov = ic_cov.loc[trade_date, :]
    
#     mat = mat_ic_cov / np.diag(mat_ic_cov).mean()
    
#     mat = mat + lambda_i * np.diag(np.ones(len(factors)))
    
#     weight.loc[trade_date, :] = np.linalg.inv(mat).dot(ic_mean.loc[trade_date, :].values)

sql = tools.generate_sql_y_x(factors, start_date, end_date)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])

y = df.loc[:, 'r_d'].unstack()

x = DataFrame(dtype='float64')
for factor in factors:
    df_x = df.loc[:, factor].unstack()
    df_x = tools.standardize(tools.winsorize(df_x))
    # df_x = df_x.rank(axis=1, pct=True)
    x = x.add(df_x.mul(weight.loc[:, factor], axis=0), fill_value=0)
# x = tools.neutralize(x, ['mc', 'bp'], ind='l3')
#因子分布
plt.figure(figsize=(16,9))
plt.hist(x.values.flatten())

#IC
IC = x.corrwith(y, axis=1)
plt.figure(figsize=(16,9))
IC.cumsum().plot()

#IR
IR = IC.rolling(20).mean() / IC.rolling(20).std()
plt.figure(figsize=(16,9))
IR.cumsum().plot()

#R2
plt.figure(figsize=(16,9))
(IC**2).cumsum().plot()

#换手
plt.figure(figsize=(16,9))
x.corrwith(x.shift(), axis=1, method='spearman').cumsum().plot()

x_quantile = DataFrame(x.rank(axis=1)).div(x.notna().sum(1), axis=0)
num_group = 21
group_pos = {}
for n in range(num_group):
    group_pos[n] = DataFrame((n/num_group <= x_quantile) & (x_quantile <= (n+1)/num_group))
    group_pos[n][~group_pos[n]] = np.nan
    group_pos[n] = 1 * group_pos[n]


from matplotlib import cm

group_mean = {}
for n in range(num_group):
    group_mean[n] = ((group_pos[n] * y).mean(1)+1).cumprod().rename('%s'%(n/num_group))
DataFrame(group_mean).plot(figsize=(16, 9), cmap='coolwarm')

long = group_pos[n] * y
short = group_pos[0] * y
long_m = long.sum().sort_values(ascending=False).dropna()
short_m = short.sum().sort_values().dropna()
long_c = long.count()
short_c = short.count()
long_w = (long>0).sum() / long_c
short_w = (short>0).sum() / short_c
long_r = long.where(long>0).mean() / -long.where(long<0).mean()
short_r = short.where(short>0).mean() / -short.where(short<0).mean()
ls_c = long_c + short_c
ls_m = long_m - short_m
ls_w = long_w - short_w
ls_r = long_r - short_r

stock_names_sql = """
select stock_code, name from tsdata.ttsstockbasic
"""
s_name = pd.read_sql(stock_names_sql, engine).set_index('stock_code').name
df_ls = DataFrame(
    {
     '多空次数':ls_c, 
     '多空收益':ls_m, 
     '多空胜率':ls_w, 
     '多空盈亏比':ls_r, 
     '多头次数':long_c, 
     '多头收益':long_m, 
     '多头胜率':long_w, 
     '多头盈亏比':long_r, 
     '空头次数':short_c, 
     '空头收益':short_m, 
     '空头胜率':short_w, 
     '空头盈亏比':short_r, 
     })
df_ls.loc[:, 'stock_name'] = [s_name[i] if i in s_name.index else i for i in df_ls.index]
df_ls = df_ls.set_index('stock_name', append=True).sort_values('多空收益', ascending=False).dropna()

group_mean = {}
for n in range(num_group):
    group_mean[n] = (group_pos[n] * y).mean(1).cumsum().rename('%s'%(n/num_group))
DataFrame(group_mean).plot(figsize=(16, 9), cmap='coolwarm')


group_mean = {}
for n in range(num_group):
    group_mean[n] = ((group_pos[n] * y).mean(1) - 1*y.mean(1)).cumsum().rename('%s'%(n/num_group))
DataFrame(group_mean).plot(figsize=(16, 9), cmap='coolwarm')

plt.figure(figsize=(16, 9))
group_hist = [group_mean[i].iloc[np.where(group_mean[i].notna())[0][-1]] for i in range(num_group)]
plt.bar(range(num_group), group_hist)

group_std = {}
for n in range(num_group):
    group_std[n] = (group_pos[n] * y).std(1).cumsum().rename('%s'%(n/num_group))
DataFrame(group_std).plot(figsize=(16, 9), cmap='coolwarm')


plt.figure(figsize=(16, 9))
group_hist = [group_std[i].iloc[np.where(group_std[i].notna())[0][-1]] for i in range(num_group)]
plt.bar(range(num_group), group_hist)

group_r = {}
for n in range(num_group):
    group_r[n] = (group_pos[n] * y).mean(1)
DataFrame(group_r).plot(kind='kde', figsize=(16, 9), cmap='coolwarm')



if __name__ == '__main__':
    pass