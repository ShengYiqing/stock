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


    
if __name__ == '__main__':
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    start_date = '20220101'
    end_date = '20230730'
    trade_date = '20230816'
    d = 'D:/stock/DataBase/StockSnapshootData/' + trade_date
    files = os.listdir(d)
    file = files[2600]
    file = 'SHSE.601288.csv'
    print(trade_date, file)
    df_tick = pd.read_csv(d+'/'+file, index_col=[0], parse_dates=[0])
    volume = df_tick.loc[df_tick.trade_type==7, 'last_volume'].resample('T').sum().replace(0, np.nan).dropna()
    n = 64
    df = DataFrame({
        'log(x)':[np.log(volume.quantile(i / n)) for i in range(1,int(n*0.382))], 
        'log(1-F(x))':[np.log(1-i/n) for i in range(1,int(n*0.382))], 
        })
    plt.scatter(df.loc[:, 'log(x)'], df.loc[:, 'log(1-F(x))'])
    plt.title(file)
    plt.ylabel('log(1-q)', fontsize=16)
    plt.xlabel('log(last_volume)', fontsize=16)