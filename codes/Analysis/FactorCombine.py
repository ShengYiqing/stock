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

lambda_i = 0.001
print('halflife_mean', halflife_mean)
print('halflife_cov', halflife_cov)

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
    'momentum': 0.01, 
    'reversal': -0.01, 
    'seasonality': 0.01, 
    'skew': 0.01, 
    'cpshl': -0.01, 
    'crhls': 0.01,
    }

# weight_sub = {}
for factor in factors:
    if factor not in weight_sub.keys():
        weight_sub[factor] = 0
weight_sub = Series(weight_sub)

start_date = '20230101'
if datetime.datetime.today().strftime('%H%M') < '2200':
    end_date = (datetime.datetime.today() - datetime.timedelta(1)).strftime('%Y%m%d')
else:
    end_date = datetime.datetime.today().strftime('%Y%m%d')

# end_date = '20190101'

print('factors: ', factors)
print('weight_sub: ', weight_sub)

trade_dates = tools.get_trade_cal(start_date, end_date)
start_date_ic = tools.trade_date_shift(start_date, 1500)

#读ic
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")

sql_ic = """
select trade_date, factor_name, 
(ic_d + rank_ic_d) / 2 as ic
from tdailyic
where factor_name in {factor_names}
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_ic = sql_ic.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date)
df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name']).loc[:, 'ic'].unstack().loc[:, factors].shift(1).fillna(method='ffill')

ic_mean = df_ic.ewm(halflife=halflife_mean, min_periods=60).mean().fillna(0)
ic_cov = df_ic.ewm(halflife=halflife_cov, min_periods=60).cov().fillna(0)
ic_corr = df_ic.ewm(halflife=halflife_cov, min_periods=60).corr().fillna(0)

weight = DataFrame(0, index=trade_dates, columns=df_ic.columns)
weight.index.name = 'trade_date'
for trade_date in trade_dates:
    mat_ic_corr = ic_corr.loc[trade_date, :]
    mat_ic_cov = ic_cov.loc[trade_date, :]
    mat = mat_ic_cov / np.diag(mat_ic_cov).mean()
    mat = mat + lambda_i * np.diag(np.ones(len(factors)))
    weight.loc[trade_date, :] = (np.linalg.inv(mat).dot(ic_mean.loc[trade_date, :].values) + weight_sub) / 2

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
tools.factor_analyse(x, y, 10, 'combine')
