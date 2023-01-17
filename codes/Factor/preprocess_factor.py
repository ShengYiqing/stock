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

def f(table, start_date, end_date):
    print(table)
    table = 'tfactor' + table
    f_list = ['mc', 'bp']
    
    sql = ' select tsb.industry industry, tf.* '
    for f in f_list:
        sql = sql + ' , t{f}.factor_value {f} '.format(f=f)
    sql = sql + ' from factor.%s tf '%table
    for f in f_list:
        sql = sql + """
        left join factor.tfactor{f} t{f} 
        on tf.stock_code = t{f}.stock_code 
        and tf.trade_date = t{f}.trade_date 
        """.format(f=f)
    sql = sql + """
    left join tsdata.ttsstockbasic tsb
    on tf.stock_code = tsb.stock_code
    """
    sql = sql + ' where tf.trade_date >= %s '%start_date
    sql = sql + ' and tf.trade_date <= %s '%end_date
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    df = pd.read_sql(sql, engine)
    def f(df):
        df.loc[:, 'PREPROCESSED_FACTOR_VALUE'] = tools.standardize(tools.winsorize(df.loc[:, 'FACTOR_VALUE']))
        df.loc[:, 'industry'] = [gc.INDUSTRY_DIC_INV[ind] for ind in df.industry]
        df.dropna(subset=['FACTOR_VALUE'], inplace=True)
        if len(df) > 0:
            X = pd.concat([pd.get_dummies(df.industry), df.loc[:, f_list]], axis=1).fillna(0)
            y = df.loc[:, 'FACTOR_VALUE']
            
            model = LinearRegression(n_jobs=-1)
            model.fit(X, y)
            y_predict = Series(model.predict(X), index=y.index)
            
            res = tools.standardize(tools.winsorize(y - y_predict))
            df.loc[:, 'NEUTRAL_FACTOR_VALUE'] = res
        return df
    df = df.groupby(by=['TRADE_DATE']).apply(f)
    df.loc[:, ['REC_CREATE_TIME', 'TRADE_DATE', 'STOCK_CODE', 'FACTOR_VALUE', 'PREPROCESSED_FACTOR_VALUE', 'NEUTRAL_FACTOR_VALUE']].to_sql(table, engine, schema='factor', if_exists='append', index=False, chunksize=5000, method=tools.mysql_replace_into)
    # with engine.connect() as conn:
    #     for i in df.index:
    #         p = df.loc[i, 'PREPROCESSED_FACTOR_VALUE']
    #         n = df.loc[i, 'NEUTRAL_FACTOR_VALUE']
    #         TRADE_DATE = df.loc[i, 'TRADE_DATE']
    #         STOCK_CODE = df.loc[i, 'STOCK_CODE']
    #         sql = """update factor.%s 
    #         set NEUTRAL_FACTOR_VALUE=%s, 
    #         PREPROCESSED_FACTOR_VALUE=%s 
    #         where stock_code = %s 
    #         and trade_date = %s """%(table, n, p, STOCK_CODE, TRADE_DATE)
    #         conn.execute(sql)
if __name__ == '__main__':
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(7)).strftime('%Y%m%d')
    # start_date = '20100101'
    trade_cal = tools.get_trade_cal(start_date, end_date) 
    files = os.listdir('D:/stock/DataBase/Data/factor')
    tables = [i.split('.')[0][7:] for i in files]
    tables = list(filter(lambda x:(x[0]!='f') & (x[0:2]!='ex') & (x!='mc') & (x!='bp') & (x!='ep') & (x!='sp') & (x!='dv') & (x!='quality') & (x!='value'), tables))
    factors = [
        'momentum', 'corrmarket', 
        'str', 
        'pvcorr', 
        ]
    tables = factors
    pool = mp.Pool(4)
    for table in tables:
        pool.apply_async(func=f, args=(table, start_date, end_date))
    pool.close()
    pool.join()
    