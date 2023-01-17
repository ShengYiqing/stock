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
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "end_type",
    "basic_eps",
    "diluted_eps",
    "total_revenue",
    "revenue",
    "int_income",
    "prem_earned",
    "comm_income",
    "n_commis_income",
    "n_oth_income",
    "n_oth_b_income",
    "prem_income",
    "out_prem",
    "une_prem_reser",
    "reins_income",
    "n_sec_tb_income",
    "n_sec_uw_income",
    "n_asset_mg_income",
    "oth_b_income",
    "fv_value_chg_gain",
    "invest_income",
    "ass_invest_income",
    "forex_gain",
    "total_cogs",
    "oper_cost",
    "int_exp",
    "comm_exp",
    "biz_tax_surchg",
    "sell_exp",
    "admin_exp",
    "fin_exp",
    "assets_impair_loss",
    "prem_refund",
    "compens_payout",
    "reser_insur_liab",
    "div_payt",
    "reins_exp",
    "oper_exp",
    "compens_payout_refu",
    "insur_reser_refu",
    "reins_cost_refund",
    "other_bus_cost",
    "operate_profit",
    "non_oper_income",
    "non_oper_exp",
    "nca_disploss",
    "total_profit",
    "income_tax",
    "n_income",
    "n_income_attr_p",
    "minority_gain",
    "oth_compr_income",
    "t_compr_income",
    "compr_inc_attr_p",
    "compr_inc_attr_m_s",
    "ebit",
    "ebitda",
    "insurance_exp",
    "undist_profit",
    "distable_profit",
    "rd_exp",
    "fin_exp_int_exp",
    "fin_exp_int_inc",
    "transfer_surplus_rese",
    "transfer_housing_imprest",
    "transfer_oth",
    "adj_lossgain",
    "withdra_legal_surplus",
    "withdra_legal_pubfund",
    "withdra_biz_devfund",
    "withdra_rese_fund",
    "withdra_oth_ersu",
    "workers_welfare",
    "distr_profit_shrhder",
    "prfshare_payable_dvd",
    "comshare_payable_dvd",
    "capit_comstock_div",
    "continued_net_profit",
    "update_flag",
    "net_after_nr_lp_correct",
    "oth_income",
    "asset_disp_income",
    "end_net_profit",
    "credit_impa_loss",
    "net_expo_hedging_benefits",
    "oth_impair_loss_assets",
    "total_opcost",
    "amodcost_fin_assets"
]
field_pk = ['STOCK_CODE', 'ANN_DATE', 'F_ANN_DATE', 'END_DATE', 'REPORT_TYPE']
field_type_dic = {'REC_CREATE_TIME': 'VARCHAR(14)',
              'STOCK_CODE': 'VARCHAR(20)',
              'TS_CODE': 'VARCHAR(20)',
              'ANN_DATE': 'VARCHAR(8)',
              'F_ANN_DATE': 'VARCHAR(8)',
              'END_DATE': 'VARCHAR(8)',
              'REPORT_TYPE': 'VARCHAR(8)',
              'COMP_TYPE': 'VARCHAR(8)',
              'END_TYPE': 'VARCHAR(8)',}
sql_create = """
CREATE TABLE 'tsdata'.'ttsincome' (
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
df1 = tools.download_tushare(pro=pro, api_name='income_vip', fields=fields, limit=5000, start_date=start_date, report_type=1)
df4 = tools.download_tushare(pro=pro, api_name='income_vip', fields=fields, limit=5000, start_date=start_date, report_type=4)
df5 = tools.download_tushare(pro=pro, api_name='income_vip', fields=fields, limit=5000, start_date=start_date, report_type=5)
df11 = tools.download_tushare(pro=pro, api_name='income_vip', fields=fields, limit=5000, start_date=start_date, report_type=11)
df = pd.concat([df1,df4,df5,df11],axis=0)
if len(df) > 0:
    # df.loc[:, 'ml'] = df.loc[:, 'revenue'].fillna(0) - df.loc[:, 'oper_cost'].fillna(0)
    # df.loc[:, 'hxlr'] = df.loc[:, 'revenue'].fillna(0) - df.loc[:, 'oper_cost'].fillna(0) - df.loc[:, 'biz_tax_surchg'].fillna(0) - df.loc[:, 'sell_exp'].fillna(0) - df.loc[:, 'admin_exp'].fillna(0) - df.loc[:, 'rd_exp'].fillna(0) - df.loc[:, 'fin_exp_int_exp'].fillna(0)
    # df.loc[:, 'hxfy'] = df.loc[:, 'oper_cost'].fillna(0) + df.loc[:, 'biz_tax_surchg'].fillna(0) + df.loc[:, 'sell_exp'].fillna(0) + df.loc[:, 'admin_exp'].fillna(0) + df.loc[:, 'rd_exp'].fillna(0) + df.loc[:, 'fin_exp_int_exp'].fillna(0) + df.loc[:, 'income_tax'].fillna(0)
    
    df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]

    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

    df.to_sql('ttsincome', engine, schema='tsdata', index=False, chunksize=1000, if_exists='append', method=tools.mysql_replace_into)
