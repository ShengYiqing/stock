
import datetime
import os
import sys
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import Global_Config as gc
import tools
from sqlalchemy import create_engine

white_threshold = 0.618
is_neutral = 0
factor_value_type = 'neutral_factor_value' if is_neutral else 'preprocessed_factor_value'

factors = [
    'value', 'quality', 
    'dailytech', 'hftech', 
    ]
weight = {'value': -1,
          'quality': 1, 
          'dailytech': 1,
          'hftech': 1,
          }
start_date = '20220101'
if datetime.datetime.today().strftime('%H%M') < '2200':
    end_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')
else:
    end_date = datetime.datetime.today().strftime('%Y%m%d')

print('factors: ', factors)

trade_dates = tools.get_trade_cal(start_date, end_date)
    
sql = tools.generate_sql_y_x(factors, start_date, end_date, white_threshold=white_threshold, factor_value_type=factor_value_type)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])

y = df.loc[:, 'r_daily'].unstack()

x = DataFrame(dtype='float64')
for factor in factors:
    print(factor)
    x = x.add(df.loc[:, factor].unstack() * weight[factor], fill_value=0)

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