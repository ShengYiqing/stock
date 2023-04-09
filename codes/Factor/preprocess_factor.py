import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import tools
import Global_Config as gc
import multiprocessing as mp
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import multiprocessing as mp
from sklearn.linear_model import LinearRegression

def f(factor, start_date, end_date):
    print(factor)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    sql = """
    select trade_date, stock_code, factor_value, preprocessed_factor_value
    from factor.tfactor{factor}
    where trade_date >= {start_date}
    and trade_date <= {end_date}
    """.format(factor=factor, start_date=start_date, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    x = df.loc[:, 'preprocessed_factor_value'].unstack()
    
    sql = """
    select tlabel.trade_date trade_date, tlabel.stock_code stock_code, tmc.preprocessed_factor_value mc, tbp.preprocessed_factor_value bp 
    from label.tdailylabel tlabel
    left join factor.tfactormc tmc
    on tlabel.stock_code = tmc.stock_code
    and tlabel.trade_date = tmc.trade_date
    left join factor.tfactorbp tbp
    on tlabel.stock_code = tbp.stock_code
    and tlabel.trade_date = tbp.trade_date
    where tlabel.trade_date in {trade_dates}
    and tlabel.stock_code in {stock_codes}
    """.format(trade_dates=tuple(x.index), stock_codes=tuple(x.columns))
    
    df_n = pd.read_sql(sql, engine)
    df_n = df_n.set_index(['trade_date', 'stock_code'])
    x = x.stack()
    x.name = 'x'
    data = pd.concat([x, df_n], axis=1).dropna()

    def g(data):
        # X = pd.concat([pd.get_dummies(data.ind), data.loc[:, ['mc', 'bp']]], axis=1).fillna(0)
        X = data.loc[:, ['mc', 'bp']]
        # print(X)
        y = data.loc[:, 'x']
        model = LinearRegression(n_jobs=-1)
        model.fit(X, y)
        y_predict = Series(model.predict(X), index=y.index)
        
        res = tools.standardize(tools.winsorize(y - y_predict))
        return res
    x_n = data.groupby('trade_date', as_index=False).apply(g).reset_index(0, drop=True)
    x_n.name = 'neutral_factor_value'
    df_n = pd.concat([df, x_n], axis=1)
    df_n.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df_n.to_sql('tfactor%s'%factor, engine, schema='factor', if_exists='append', index=True, chunksize=5000, method=tools.mysql_replace_into)

if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
    # start_date = '20100101'
    trade_cal = tools.get_trade_cal(start_date, end_date) 
    factors = [
        'operation', 
        'profitability', 
        'growth', 
        ]
    pool = mp.Pool(4)
    for factor in factors:
        pool.apply_async(func=f, args=(factor, start_date, end_date))
    pool.close()
    pool.join()
    