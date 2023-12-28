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

#%%
start_date = '20120901'
end_date = '20230930'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")

sql_y = tools.generate_sql_y_x([], start_date, end_date)
df_y = pd.read_sql(sql_y, engine)
y = df_y.set_index(['trade_date', 'stock_code']).r_d.unstack()
stock_codes = tuple(y.columns)
#%%
start_date_sql = tools.trade_date_shift(start_date, 750)
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")

table_fin_dic = {
    'ttsbalancesheet': ['total_hldr_eqy_inc_min_int', ], 
    'ttsincome': ['revenue', ], 
    }
for table in table_fin_dic.keys():
    for fin in table_fin_dic[table]:
        sql = """
        select end_date, stock_code, {fin}
        from tsdata.{table}
        where f_ann_date >= {start_date}
        and f_ann_date <= {end_date}
        """.format(table=table, fin=fin, start_date=start_date_sql, end_date=end_date)
        df_sql = pd.read_sql(sql, engine)
sql = """
select ann_date, end_date, stock_code, financial_index, financial_value 
from findata.tfindata
where end_date >= {start_date}
and end_date <= {end_date}
and substr(end_date, -4) in ('0331', '0630', '0930', '1231')
and ann_type in ('定期报告')
and financial_index in ('zzc', 'yysr')
""".format(start_date=start_date_sql, end_date=end_date)
df_sql = pd.read_sql(sql, engine).sort_values(['financial_index', 'end_date', 'stock_code', 'ann_date'])

trade_dates = tools.get_trade_cal(start_date, end_date)

factor = DataFrame()
dic = {}
for trade_date in trade_dates:
    print(trade_date)
    start_date_tmp = tools.trade_date_shift(trade_date, 750)
    df_tmp = df_sql.loc[(df_sql.ann_date>=start_date_tmp)&(df_sql.ann_date<=trade_date)]
    df_tmp = df_tmp.groupby(['financial_index', 'end_date', 'stock_code']).last()
    
    yysr = df_tmp.loc['yysr'].loc[:, 'financial_value'].unstack()
    cols = yysr.columns
    yysr['YYYY'] = [ind[:4] for ind in yysr.index]
    yysr = yysr.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
    yysr = yysr.loc[:, cols]
    
    zzc = df_tmp.loc['zzc'].loc[:, 'financial_value'].unstack()
    zzc[zzc<=0] = np.nan
    zzc = zzc.rolling(2, min_periods=1).mean()
    
    yysr_ttm = yysr.rolling(4, min_periods=1).mean()
    zzc_ttm = zzc.rolling(4, min_periods=1).mean()
    
    operation = yysr_ttm / zzc_ttm
    operation.fillna(method='ffill', limit=4, inplace=True)
    dic[trade_date] = operation.iloc[-1]
x = DataFrame(dic).T
# x.index.name = 'trade_date'
# x.columns.name = 'stock_code'
# x = tools.neutralize(x)
x_ = DataFrame(x, index=y.index, columns=y.columns)
x_[y.isna()] = np.nan
tools.factor_analyse(x_, y, 5, 'operation')
