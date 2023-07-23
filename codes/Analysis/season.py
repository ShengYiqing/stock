import os
import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import datetime
import Global_Config as gc
import tools
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import tools
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

start_date = '20120101'
end_date = '20230101'
sql = """
select trade_date, factor_name, (ic_d + rank_ic_d)/2 ic_d
from factorevaluation.tdailyic
where trade_date >= {start_date}
and trade_date <= {end_date}
""".format(start_date=start_date, end_date=end_date)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306")

sql = """
select trade_date, close, open
from tsdata.ttsindexdaily
"""
df_sql = pd.read_sql(sql, engine).set_index(['trade_date'])
df = DataFrame()
df.loc[:, 'intraday'] = df_sql.close / df_sql.open - 1
df.loc[:, 'interday'] = df_sql.open / df_sql.close.shift() - 1

df.index = pd.to_datetime(df.index)
factors = df.columns
df.loc[:, 'weekday'] = [i.weekday()+1 for i in df.index]
df.loc[:, 'year'] = [i.year for i in df.index]
df = df.set_index(['year', 'weekday'])
df_tmp_dic = {factor:DataFrame(index=range(1, 6), columns=range(2014, 2024)) for factor in factors}


for factor in factors:
    plt.figure(figsize=(16, 9))
    for year in range(2014, 2024):
        s = df.loc[year, :].loc[:, factor].groupby('weekday').mean()
        df_tmp_dic[factor].loc[:, year] = s
        s.plot()
    plt.legend(range(2014, 2024))
    plt.title(factor)
    
    plt.figure(figsize=(16, 9))
    df_tmp_dic[factor].T.boxplot()
    plt.title(factor)
    
