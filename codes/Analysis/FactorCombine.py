
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

halflife_mean = 20
halflife_cov = 60

seasonal_n_mean = 20
seasonal_n_cov = 20
s = 10
lambda_i = 0.01
print('halflife_mean', halflife_mean)
print('halflife_cov', halflife_cov)
print('seasonal_n_mean', seasonal_n_mean)
print('seasonal_n_cov', seasonal_n_cov)
print('s', s)

factors = [
    'quality', 
    'momentum', 'volatility', 'speculation', 
    'dailytech', 'hftech', 
    ]
factor_value_type_dic = {factor: 'neutral_factor_value' for factor in factors}

ic_sub = {'mc':0.01, 'bp':0.01}
# ic_sub = {}

for factor in factors:
    if factor not in ic_sub.keys():
        ic_sub[factor] = 0
ic_sub = Series(ic_sub)

start_date = '20120101'
if datetime.datetime.today().strftime('%H%M') < '2200':
    end_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')
else:
    end_date = datetime.datetime.today().strftime('%Y%m%d')

# end_date = '20230302'

print('halflife_mean: ', halflife_mean)
print('halflife_cov: ', halflife_cov)
print('factors: ', factors)
print('ic_sub: ', ic_sub)

trade_dates = tools.get_trade_cal(start_date, end_date)
start_date_ic = tools.trade_date_shift(start_date, 1250)

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

n_1 = 250 - seasonal_n_mean
n_2 = n_1 + 250
n_3 = n_2 + 250
n_4 = n_3 + 250
weight_dic = {}
for t in ic_dic.keys():
    df_ic = ic_dic[t]
    df_h = h_dic[t]
    df_tr = tr_dic[t]
    
    ic_mean = df_ic.ewm(halflife=halflife_mean, min_periods=5).mean().fillna(0)
    ic_mean_s_1 = df_ic.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_1)
    ic_mean_s_2 = df_ic.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_2)
    ic_mean_s_3 = df_ic.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_3)
    ic_mean_s_4 = df_ic.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_4)
    ic_mean = 10*ic_mean + 4*ic_mean_s_1 + 3*ic_mean_s_2 + 2*ic_mean_s_3 + ic_mean_s_4
    ic_mean = ic_mean / 20
    
    ic_std = df_ic.ewm(halflife=halflife_cov, min_periods=5).std().fillna(0)
    ic_std_s_1 = df_ic.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').std(std=s).fillna(0).shift(n_1)
    ic_std_s_2 = df_ic.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').std(std=s).fillna(0).shift(n_2)
    ic_std_s_3 = df_ic.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').std(std=s).fillna(0).shift(n_3)
    ic_std_s_4 = df_ic.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').std(std=s).fillna(0).shift(n_4)
    ic_std = 10*ic_std + 4*ic_std_s_1 + 3*ic_std_s_2 + 2*ic_std_s_3 + ic_std_s_4
    ic_std = ic_std / 20
    
    ic_corr = df_ic.ewm(halflife=halflife_cov, min_periods=5).corr().fillna(0)
    ic_corr_s_1 = df_ic.rolling(seasonal_n_cov, min_periods=5).corr().fillna(0).shift(n_1*len(factors))
    ic_corr_s_2 = df_ic.rolling(seasonal_n_cov, min_periods=5).corr().fillna(0).shift(n_2*len(factors))
    ic_corr_s_3 = df_ic.rolling(seasonal_n_cov, min_periods=5).corr().fillna(0).shift(n_3*len(factors))
    ic_corr_s_4 = df_ic.rolling(seasonal_n_cov, min_periods=5).corr().fillna(0).shift(n_4*len(factors))
    ic_corr = 10*ic_corr + 4*ic_corr_s_1 + 3*ic_corr_s_2 + 2*ic_corr_s_3 + ic_corr_s_4
    ic_corr = ic_corr / 20
    
    h_mean = df_h.ewm(halflife=halflife_mean, min_periods=5).mean().fillna(0)
    h_mean_s_1 = df_h.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_1)
    h_mean_s_2 = df_h.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_2)
    h_mean_s_3 = df_h.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_3)
    h_mean_s_4 = df_h.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_4)
    h_mean = 10*h_mean + 4*h_mean_s_1 + 3*h_mean_s_2 + 2*h_mean_s_3 + h_mean_s_4
    h_mean = h_mean / 20
    
    tr_mean = df_tr.ewm(halflife=halflife_mean, min_periods=5).mean().fillna(0)
    tr_mean_s_1 = df_tr.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_1)
    tr_mean_s_2 = df_tr.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_2)
    tr_mean_s_3 = df_tr.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_3)
    tr_mean_s_4 = df_tr.rolling(seasonal_n_mean, min_periods=5, win_type='gaussian').mean(std=s).fillna(0).shift(n_4)
    tr_mean = 10*tr_mean + 4*tr_mean_s_1 + 3*tr_mean_s_2 + 2*tr_mean_s_3 + tr_mean_s_4
    tr_mean = tr_mean / 20
    
    weight = DataFrame(0, index=trade_dates, columns=df_ic.columns)
    weight.index.name = 'trade_date'
    for trade_date in trade_dates:
        mat_ic_corr = ic_corr.loc[trade_date, :]
        mat_ic_corr_tune = mat_ic_corr ** 3
        
        ic_s = ic_std.loc[trade_date, :]
        
        h = h_mean.loc[trade_date, :] ** 0.25
        tr = tr_mean.loc[trade_date, :] ** 0.25
        mat_ic_s_tune = np.diag(ic_s * h)
        
        mat_ic_cov = mat_ic_s_tune.dot(mat_ic_corr_tune).dot(mat_ic_s_tune)
        mat = mat_ic_cov / np.diag(mat_ic_cov).mean()
        mat = mat + lambda_i * np.diag(np.ones(len(factors)))
        weight.loc[trade_date, :] = np.linalg.inv(mat).dot((ic_mean.loc[trade_date, :] * tr).values)
    
    weight_dic[t] = weight.div(weight.std(1), axis=0)
weight_dic['d'] = 3 * weight_dic['d']
weight_dic['w'] = 2 * weight_dic['w']
weight_dic['m'] = 1 * weight_dic['m']
weight = pd.concat([weight.stack() for weight in weight_dic.values()], axis=1).mean(1).unstack()

sql = tools.generate_sql_y_x(factors, start_date, end_date, factor_value_type_dic=factor_value_type_dic, y_neutral=False)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])

y = df.loc[:, 'r_daily'].unstack()

x = DataFrame(dtype='float64')
for factor in factors:
    x = x.add((df.loc[:, factor].unstack().mul(weight.loc[:, factor], axis=0)), fill_value=0)


sql = """
select tlabel.trade_date trade_date, tlabel.stock_code stock_code, tind.ind_code ind, tmc.preprocessed_factor_value mc, tbp.preprocessed_factor_value bp 
from label.tdailylabel tlabel
left join indsw.tindsw tind
on tlabel.stock_code = tind.stock_code
left join factor.tfactormc tmc
on tlabel.stock_code = tmc.stock_code
and tlabel.trade_date = tmc.trade_date
left join factor.tfactorbp tbp
on tlabel.stock_code = tbp.stock_code
and tlabel.trade_date = tbp.trade_date
where tlabel.trade_date in {trade_dates}
and tlabel.stock_code in {stock_codes}""".format(trade_dates=tuple(x.index), stock_codes=tuple(x.columns))
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
df_n = pd.read_sql(sql, engine)
df_n = df_n.set_index(['trade_date', 'stock_code'])
x = x.stack()
x.name = 'x'
data = pd.concat([x, df_n], axis=1).dropna()

def f(data):
    # pdb.set_trace()
    X = pd.concat([pd.get_dummies(data.ind), data.loc[:, ['mc', 'bp']]], axis=1).fillna(0)
    # X = data.loc[:, ['mc', 'bp']]
    # print(X)
    y = data.loc[:, 'x']
    # model = LinearRegression(n_jobs=-1)
    # model.fit(X, y)
    # y_predict = Series(model.predict(X), index=y.index)
    y_predict = X.dot(np.linalg.inv(X.T.dot(X)+0.01*np.identity(len(X.T))).dot(X.T).dot(y))
    
    res = y - y_predict
    return res
x_n = data.groupby('trade_date').apply(f).unstack()
x_n.reset_index(0, drop=True, inplace=True)
x = x_n

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

plt.figure(figsize=(16, 12))
for n in range(num_group):
    (group_pos[n] * tools.winsorize(y)).stack().plot(kind='kde')
    plt.legend(['%s'%i for i in range(num_group)], loc="best")


if __name__ == '__main__':
    pass