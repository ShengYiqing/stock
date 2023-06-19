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

def single_factor_analysis(factor_name, start_date, end_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    
    sql = tools.generate_sql_y_x([factor_name], start_date, end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    
    x = df.loc[:, factor_name].unstack()
    # x = tools.neutralize(x)
    y = df.loc[:, 'r_daily'].unstack()
    
    tools.factor_analyse(x, y, 10, factor_name)
    
    
if __name__ == '__main__':
    # factor_name = 'operation'
    factors = [
        'quality', 
        'operation', 'gross', 
        'core', 'profitability', 'cash', 
        'growth', 'stability'
        ]
    factors = ['beta']
    trade_date = '20230613'
    
    print(factors, trade_date)
    
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
    
    sql = """
    select 
    tl.is_trade, 
    tind.l1_name, tind.l2_name, tind.l3_name, 
    tmc.stock_code, tsb.name, 
    tmc.factor_value mc
    """
    for factor in factors:
        sql += """
        , t{factor}.factor_value {factor}
        """.format(factor=factor)
    sql += """
    from factor.tfactormc tmc
    left join tsdata.ttsstockbasic tsb
    on tmc.stock_code = tsb.stock_code
    left join indsw.tindsw tind
    on tmc.stock_code = tind.stock_code
    left join label.tdailylabel tl
    on tmc.stock_code = tl.stock_code
    and tmc.trade_date = tl.trade_date
    """
    for factor in factors:
        sql += """
        left join factor.tfactor{factor} t{factor}
        on tmc.stock_code = t{factor}.stock_code
        and tmc.trade_date = t{factor}.trade_date
        """.format(factor=factor)
    sql += """
    where tmc.trade_date = {trade_date}
    and tind.l3_name in {white_ind}
    order by tind.l1_name, tind.l2_name, tind.l3_name, tmc.factor_value desc
    """.format(trade_date=trade_date, white_ind=tuple(gc.WHITE_INDUSTRY_LIST))
    # sql += """
    # where tmc.trade_date = {trade_date}
    # order by tind.l1_name, tind.l2_name, tind.l3_name, tmc.factor_value desc
    # """.format(trade_date=trade_date)
    df = pd.read_sql(sql, engine)
    df.loc[:, ['mc']+factors] = df.loc[:, ['mc']+factors].rank(pct=True)
