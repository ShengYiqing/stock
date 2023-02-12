
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

white_threshold = 0.8
is_neutral = 0
factor_value_type = 'neutral_factor_value' if is_neutral else 'preprocessed_factor_value'
halflife_ic_mean = 250
halflife_ic_cov = 750
lambda_ic = 50
lambda_i = 50

factors = [
    'value', 'quality', 
    'momentum', 'corrmarket', 
    'str', 
    'pvcorr', 
    ]

ic_sub = {'mc':0.01, 'bp':0.01}
ic_sub = {}

for factor in factors:
    if factor not in ic_sub.keys():
        ic_sub[factor] = 0
ic_sub = Series(ic_sub)

start_date = '20200101'
if datetime.datetime.today().strftime('%H%M') < '2200':
    end_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')
else:
    end_date = datetime.datetime.today().strftime('%Y%m%d')

# end_date = datetime.datetime.today().strftime('%Y%m%d')

print('halflife_ic_mean: ', halflife_ic_mean)
print('halflife_ic_cov: ', halflife_ic_cov)
print('lambda_ic: ', lambda_ic)
print('lambda_i: ', lambda_i)
print('factors: ', factors)
print('ic_sub: ', ic_sub)

trade_dates = tools.get_trade_cal(start_date, end_date)
start_date_ic = tools.trade_date_shift(start_date, 800)

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
sql_ic = sql_ic.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date, white_threshold=white_threshold, is_neutral=is_neutral)
df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name'])
df_ic_m = df_ic.loc[:, 'ic_m'].unstack().loc[:, factors].shift(21).fillna(method='ffill')
df_ic_w = df_ic.loc[:, 'ic_w'].unstack().loc[:, factors].shift(6).fillna(method='ffill')
df_ic_d = df_ic.loc[:, 'ic_d'].unstack().loc[:, factors].shift(2).fillna(method='ffill')
df_ic = pd.concat([df_ic_m.stack(), df_ic_w.stack(), df_ic_d.stack()], axis=1).mean(1).unstack()
df_ic = df_ic_d
ic_mean = df_ic.ewm(halflife=halflife_ic_mean).mean().fillna(0)
ic_cov = df_ic.ewm(halflife=halflife_ic_cov).cov().fillna(0)

weight = DataFrame(0, index=trade_dates, columns=df_ic.columns)
for trade_date in trade_dates:
    mat_ic = ic_cov.loc[trade_date, :].values
    mat_ic = mat_ic / np.trace(mat_ic)
    mat_i = np.diag(np.ones(len(factors)))
    mat_i = mat_i / np.trace(mat_i)
    mat = lambda_ic * mat_ic + lambda_i * mat_i
    weight.loc[trade_date, :] = np.linalg.inv(mat).dot(ic_mean.loc[trade_date, :].values / 2)
    
    
sql = tools.generate_sql_y_x(factors, start_date, end_date, white_threshold=white_threshold, factor_value_type=factor_value_type)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])

y = df.loc[:, 'r_daily'].unstack()

x = DataFrame(dtype='float64')
for factor in factors:
    print(factor)
    x = x.add((df.loc[:, factor].unstack().mul(weight.loc[:, factor], axis=0)), fill_value=0)

#因子分布
plt.figure(figsize=(16,12))
plt.hist(x.values.flatten())
# plt.savefig('%s/Results/%s/hist.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))

#IC
IC = x.corrwith(y, axis=1)
plt.figure(figsize=(16,12))
IC.cumsum().plot()
# plt.savefig('%s/Results/%s/IC.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))

#IR
IR = IC.rolling(20).mean() / IC.rolling(20).std()
plt.figure(figsize=(16,12))
IR.cumsum().plot()
# plt.savefig('%s/Results/%s/IR.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))

#ICabs
plt.figure(figsize=(16,12))
IC.abs().cumsum().plot()
# plt.savefig('%s/Results/%s/IC_abs.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))

#换手
plt.figure(figsize=(16,12))
x.corrwith(x.shift(), axis=1, method='spearman').cumsum().plot()

x_quantile = DataFrame(x.rank(axis=1)).div(x.notna().sum(1), axis=0)
num_group = 10
group_pos = {}
for n in range(num_group):
    group_pos[n] = DataFrame((n/num_group <= x_quantile) & (x_quantile <= (n+1)/num_group))
    group_pos[n][~group_pos[n]] = np.nan
    group_pos[n] = 1 * group_pos[n]
        
plt.figure(figsize=(16, 12))
group_mean = {}
for n in range(num_group):
    group_mean[n] = ((group_pos[n] * y).mean(1)+1).cumprod().rename('%s'%(n/num_group))
    group_mean[n].plot()
plt.legend(['%s'%i for i in range(num_group)], loc="best")
# plt.savefig('%s/Results/%s/group_mean%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))

plt.figure(figsize=(16, 12))
group_mean = {}
for n in range(num_group):
    group_mean[n] = (group_pos[n] * y).mean(1).cumsum().rename('%s'%(n/num_group))
    group_mean[n].plot()
plt.legend(['%s'%i for i in range(num_group)], loc="best")
# plt.savefig('%s/Results/%s/group_mean%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))

plt.figure(figsize=(16, 12))
group_mean = {}
for n in range(num_group):
    group_mean[n] = ((group_pos[n] * y).mean(1) - 1*y.mean(1)).cumsum().rename('%s'%(n/num_group))
    group_mean[n].plot()
plt.legend(['%s'%i for i in range(num_group)], loc="best")
# plt.savefig('%s/Results/%s/group_mean%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))

plt.figure(figsize=(16, 12))
group_hist = [group_mean[i].iloc[np.where(group_mean[i].notna())[0][-1]] for i in range(num_group)]
plt.bar(range(num_group), group_hist)
# plt.savefig('%s/Results/%s/group_mean_hist%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))

plt.figure(figsize=(16, 12))
group_std = {}
for n in range(num_group):
    group_std[n] = (group_pos[n] * y).std(1).cumsum().rename('%s'%(n/num_group))
    group_std[n].plot()
plt.legend(['%s'%i for i in range(num_group)], loc="best")
# plt.savefig('%s/Results/%s/group_std%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))

plt.figure(figsize=(16, 12))
group_hist = [group_std[i].iloc[np.where(group_std[i].notna())[0][-1]] for i in range(num_group)]
plt.bar(range(num_group), group_hist)
# plt.savefig('%s/Results/%s/group_std_hist%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))


if __name__ == '__main__':
    pass