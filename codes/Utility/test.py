import os
import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import Global_Config as gc
import tools
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import tools
sql = """
select trade_date, stock_code, r_daily
from label.tdailylabel
where trade_date >= '20200101'
and trade_date <= '20230101'
"""
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
df = df.unstack()
benchmark = df.mean(1).cumsum()
timing = df.mean(1).abs().cumsum()

plt.figure(figsize=(16, 9))
benchmark.plot()
timing.plot()

d_q = df[df.ge(df.quantile(0.9, axis=1), axis=0)].mean(1)


sql = """
SELECT t1.factor_value mc, t2.factor_value bp
FROM factor.tfactormc t1
left join factor.tfactorbp t2
on t1.trade_date = t2.trade_date
and t1.stock_code = t2.stock_code
where t1.trade_date = '20230407'
"""
sql = tools.generate_sql_y_x(['mc', 'bp'], '20230407', '20230407', white_threshold=0.2, factor_value_type='factor_value')
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
df = pd.read_sql(sql, engine)
mc = df.mc.rank(pct=True)
bp = df.bp.rank(pct=True)
plt.scatter(mc, bp)
df_tmp = DataFrame(index=range(1, 4), columns=range(1, 4))
df_tmp.index.name = 'mc'
df_tmp.columns.name = 'bp'

for i in range(1, 4):
    for j in range(1, 4):
        mc_mask = ((i-1)/3<=mc) & (mc<=i/3)
        bp_mask = ((j-1)/3<=bp) & (bp<=j/3)
        df_tmp.loc[i, j] = (mc_mask & bp_mask).mean()
df = DataFrame(np.arange(80).reshape(20, 4))
df.rolling(10, win_type='exponential').mean()

trade_date = '20220104'
x1 = 'corrmarket'
x2 = 'hfcorrmarket'
def dailyhffactor_analysis(x1, x2, trade_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    
    sql = """
    select stock_code, factor_name , factor_value from intermediate.tdailyhffactor
    where trade_date = {trade_date}
    and factor_name in ('{x1}', '{x2}')
    """.format(x1=x1, x2=x2, trade_date=trade_date)
    df = pd.read_sql(sql, engine).set_index(['stock_code', 'factor_name']).unstack()
    df.dropna(inplace=True)
    plt.scatter(df.iloc[:, 0].rank(), df.iloc[:, 1].rank())
    print(df.corr())
dailyhffactor_analysis(x1, x2, trade_date)

for ind_k in gc.WHITE_INDUSTRY_DIC.keys():
    if (len(gc.WHITE_INDUSTRY_DIC[ind_k]) > 0):
        sql = """
        select count(1) from tsdata.ttsstockbasic where industry in %s
        """%gc.WHITE_INDUSTRY_DIC[ind_k]
        sql = sql.replace('[', '(').replace(']', ')')
        engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
        df = pd.read_sql(sql, engine)
        print(ind_k, df.iloc[0,0])
def quality_analyse(stock_code):
    files = os.listdir('D:/stock/DataBase/Data/factor')
    tables = [i.split('.')[0] for i in files]
    tables = list(filter(lambda x:x[7]=='f', tables))
    
    sql = """ select tfactorquality.trade_date, tfactorquality.stock_code, tfactorquality.preprocessed_factor_value quality """
    for table in tables:
        sql += ' , %s.preprocessed_factor_value %s'%(table, table[7:])
    sql += ' from tfactorquality '
    for table in tables:
        sql += ' left join %s on tfactorquality.trade_date = %s.trade_date and tfactorquality.stock_code = %s.stock_code '%(table, table, table)
    sql += """where tfactorquality.trade_date = '20230103' and tfactorquality.stock_code in %s"""%stock_code
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df = pd.read_sql(sql, engine)
    return df.T
stock_code = '(002755)'
quality_analyse(stock_code)


files = os.listdir('D:/stock/DataBase/Data/factor')
tables = [i.split('.')[0] for i in files]
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
tables = ['tfactoramount']
for table in tables:
    print(table)
    sql = """optimize table factor.%s"""%table
    with engine.connect() as conn:
        conn.execute(sql)


trade_date = '20221212'
x1 = 'profitability'
x2 = 'growth'
tools.colinearity_analysis(x1, x2, trade_date)
