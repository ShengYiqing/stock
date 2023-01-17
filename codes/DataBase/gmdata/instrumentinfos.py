# coding=utf-8
from __future__ import print_function, absolute_import
import gm.api.basic as basic
import gm.api.query as query
import Config
import sys
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools

import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import multiprocessing as mp
import os
from sqlalchemy import create_engine


if __name__ == '__main__':

    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/gmdata?charset=utf8")
    f = open("%s/gm.txt"%gc.PROJECT_PATH)
    s = f.read()
    f.close()
    token = s
    basic.set_token(token)

    df = query.get_instrumentinfos(sec_types='1', fields='symbol, sec_id, sec_name, exchange, listed_date, delisted_date', df=True)
    
    df.loc[:, 'listed_date'] = [i.strftime('%Y%m%d') for i in df.listed_date]
    df.loc[:, 'delisted_date'] = [i.strftime('%Y%m%d') for i in df.delisted_date]
    
    df.rename(columns={'exchange': 'exchange_code', 'sec_id': 'stock_code', 'sec_name': 'name'}, inplace=True)
    
    df.loc[:, 'REC_CREATOR'] = 'gm'
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    df.loc[:, 'REC_REVISOR'] = 'gm'
    df.loc[:, 'REC_REVISE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    
    df.to_sql('tgminstrumentinfos', engine, schema='gmdata', index=False, if_exists='append', method=tools.mysql_replace_into)
