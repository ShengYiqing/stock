import tushare as ts
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import Global_Config as gc
import tools
import sqlalchemy as sa
from sqlalchemy import create_engine

fields=[
    "ts_code",
    "ann_date",
    "end_date",
    "revenue",
    "operate_profit",
    "total_profit",
    "n_income",
    "total_assets",
    "total_hldr_eqy_exc_min_int",
    "diluted_eps",
    "diluted_roe",
    "yoy_net_profit",
    "bps",
    "perf_summary",
    "yoy_sales",
    "yoy_op",
    "yoy_tp",
    "yoy_dedu_np",
    "yoy_eps",
    "yoy_roe",
    "growth_assets",
    "yoy_equity",
    "growth_bps",
    "or_last_year",
    "op_last_year",
    "tp_last_year",
    "np_last_year",
    "eps_last_year",
    "open_net_assets",
    "open_bps",
    "is_audit",
    "remark"
]

field_pk = ['STOCK_CODE', 'ANN_DATE', 'END_DATE']
field_type_dic = {'REC_CREATE_TIME': 'VARCHAR(14)',
              'STOCK_CODE': 'VARCHAR(20)',
              'TS_CODE': 'VARCHAR(20)',
              'ANN_DATE': 'VARCHAR(8)',
              'END_DATE': 'VARCHAR(8)',}
sql_create = """
CREATE TABLE 'tsdata'.'ttsexpress' (
    'REC_CREATE_TIME' VARCHAR(14) NULL,
    'STOCK_CODE' VARCHAR(20) NOT NULL,
"""
for field in fields:
    field = field.upper()
    field_type = 'DOUBLE'
    if field in field_type_dic.keys():
        field_type = field_type_dic[field]
    null = 'NULL'
    if field in field_pk:
        null = 'NOT NULL'
    sql_create = sql_create + """ \'{field}\' {field_type} {null}, """.format(field=field, field_type=field_type, null=null)
sql_create = sql_create + """ PRIMARY KEY ( """
for pk in field_pk:
    sql_create = sql_create + """\'{pk}\',""".format(pk=pk)
sql_create = sql_create[:-1] + '))'

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
# engine.execute(sql_create)

pro = ts.pro_api()

start_date = str(int(datetime.datetime.today().strftime('%Y')) - 1) + '0101'
# start_date = '20000101'
# sql = """
# select ts_code from ttsstockbasic
# """
# stock_list = list(pd.read_sql(sql, engine).loc[:, 'ts_code'])
# for stock in stock_list:
df = tools.download_tushare(pro=pro, api_name='express_vip', fields=fields, limit=5000, start_date=start_date)
if len(df) > 0:
    df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]
    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

    df.to_sql('ttsexpress', engine, schema='tsdata', index=False, chunksize=1000, if_exists='append', method=tools.mysql_replace_into)
