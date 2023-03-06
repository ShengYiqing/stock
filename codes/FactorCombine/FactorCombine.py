
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
n_ic_mean = 60
n_ic_cov = 250
lambda_ic = 50
lambda_i = 50

factor_dic = {
    'dailytech':['momentum', 'sigma', 'skew', 'hl', 
                 'corrmarket', 'corrmarketa', 
                 'tr', 'str', 
                 'pvcorr',], 
    'hftech':['hfcallauctionmomentum', 'hfintradaymomentum', 
              'hfhl', 'hfsigma', 'hfskew', 
              'hfcorrmarket', 
              'hfetr', 'hfttr', 'hfutr', 
              'hfpicorr', 'hfpscorr', 
              'hfspread',], 
    }
factors = sum(list(factor_dic.values()), [])
end_date = datetime.datetime.today().strftime('%Y%m%d')
start_date = (datetime.datetime.today() - datetime.timedelta(7)).strftime('%Y%m%d')
# start_date = '20100101'
trade_dates = tools.get_trade_cal(start_date, end_date)
start_date_ic = tools.trade_date_shift(start_date, 800)

#读ic
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/ic?charset=utf8")

sql_ic = """
select trade_date, factor_name, 
(ic_d+rank_ic_d)/2 as ic
from tdailyic
where factor_name in {factor_names}
and trade_date >= {start_date}
and trade_date <= {end_date}
and white_threshold = {white_threshold}
and is_neutral = {is_neutral}
"""
sql_ic = sql_ic.format(factor_names='(\''+'\',\''.join(factors)+'\')', start_date=start_date_ic, end_date=end_date, white_threshold=white_threshold, is_neutral=is_neutral)
df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name'])
df_ic = df_ic.loc[:, 'ic'].unstack().loc[:, factors].shift(2).fillna(method='ffill')

ic_mean = df_ic.rolling(n_ic_mean).mean().fillna(0)
ic_cov = df_ic.rolling(n_ic_cov).cov().fillna(0)

weight = DataFrame(0, index=trade_dates, columns=df_ic.columns)
for trade_date in trade_dates:
    mat_ic = ic_cov.loc[trade_date, :].values
    mat_ic = mat_ic / np.trace(mat_ic)
    mat_i = np.diag(np.ones(len(factors)))
    mat_i = mat_i / np.trace(mat_i)
    mat = lambda_ic * mat_ic + lambda_i * mat_i
    weight.loc[trade_date, :] = np.linalg.inv(mat).dot(ic_mean.loc[trade_date, :].values / 2)
    

sql = tools.generate_sql_y_x(factors, start_date, end_date, white_threshold=0, factor_value_type=factor_value_type)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])

for factor_1 in factor_dic.keys():
    
    x = DataFrame(dtype='float64')
    for factor_2 in factor_dic[factor_1]:
        x = x.add((df.loc[:, factor_2].unstack().mul(weight.loc[:, factor_2], axis=0)), fill_value=0)
        df_p = tools.standardize(tools.winsorize(x))
        df_new = pd.concat([x, df_p], axis=1, keys=['FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE'])
        df_new = df_new.stack()
        df_new.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
        df_new.to_sql('tfactor%s'%factor_1, engine, schema='factor', if_exists='append', index=True, chunksize=10000, method=tools.mysql_replace_into)

if __name__ == '__main__':
    pass