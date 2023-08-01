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
    sql = tools.generate_sql_y_x([], start_date, end_date)
    df = pd.read_sql(sql, engine)
    df = df.set_index(['trade_date', 'stock_code']).my_rank.unstack()
    ((df * df.shift()).notna().sum(1) / 168).plot()
    df.corrwith(df.shift(), axis=1).plot()
    