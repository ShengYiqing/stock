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
    "total_share",
    "cap_rese",
    "undistr_porfit",
    "surplus_rese",
    "special_rese",
    "money_cap",
    "trad_asset",
    "notes_receiv",
    "accounts_receiv",
    "oth_receiv",
    "prepayment",
    "div_receiv",
    "int_receiv",
    "inventories",
    "amor_exp",
    "nca_within_1y",
    "sett_rsrv",
    "loanto_oth_bank_fi",
    "premium_receiv",
    "reinsur_receiv",
    "reinsur_res_receiv",
    "pur_resale_fa",
    "oth_cur_assets",
    "total_cur_assets",
    "fa_avail_for_sale",
    "htm_invest",
    "lt_eqt_invest",
    "invest_real_estate",
    "time_deposits",
    "oth_assets",
    "lt_rec",
    "fix_assets",
    "cip",
    "const_materials",
    "fixed_assets_disp",
    "produc_bio_assets",
    "oil_and_gas_assets",
    "intan_assets",
    "r_and_d",
    "goodwill",
    "lt_amor_exp",
    "defer_tax_assets",
    "decr_in_disbur",
    "oth_nca",
    "total_nca",
    "cash_reser_cb",
    "depos_in_oth_bfi",
    "prec_metals",
    "deriv_assets",
    "rr_reins_une_prem",
    "rr_reins_outstd_cla",
    "rr_reins_lins_liab",
    "rr_reins_lthins_liab",
    "refund_depos",
    "ph_pledge_loans",
    "refund_cap_depos",
    "indep_acct_assets",
    "client_depos",
    "client_prov",
    "transac_seat_fee",
    "invest_as_receiv",
    "total_assets",
    "lt_borr",
    "st_borr",
    "cb_borr",
    "depos_ib_deposits",
    "loan_oth_bank",
    "trading_fl",
    "notes_payable",
    "acct_payable",
    "adv_receipts",
    "sold_for_repur_fa",
    "comm_payable",
    "payroll_payable",
    "taxes_payable",
    "int_payable",
    "div_payable",
    "oth_payable",
    "acc_exp",
    "deferred_inc",
    "st_bonds_payable",
    "payable_to_reinsurer",
    "rsrv_insur_cont",
    "acting_trading_sec",
    "acting_uw_sec",
    "non_cur_liab_due_1y",
    "oth_cur_liab",
    "total_cur_liab",
    "bond_payable",
    "lt_payable",
    "specific_payables",
    "estimated_liab",
    "defer_tax_liab",
    "defer_inc_non_cur_liab",
    "oth_ncl",
    "total_ncl",
    "depos_oth_bfi",
    "deriv_liab",
    "depos",
    "agency_bus_liab",
    "oth_liab",
    "prem_receiv_adva",
    "depos_received",
    "ph_invest",
    "reser_une_prem",
    "reser_outstd_claims",
    "reser_lins_liab",
    "reser_lthins_liab",
    "indept_acc_liab",
    "pledge_borr",
    "indem_payable",
    "policy_div_payable",
    "total_liab",
    "treasury_share",
    "ordin_risk_reser",
    "forex_differ",
    "invest_loss_unconf",
    "minority_int",
    "total_hldr_eqy_exc_min_int",
    "total_hldr_eqy_inc_min_int",
    "total_liab_hldr_eqy",
    "lt_payroll_payable",
    "oth_comp_income",
    "oth_eqt_tools",
    "oth_eqt_tools_p_shr",
    "lending_funds",
    "acc_receivable",
    "st_fin_payable",
    "payables",
    "hfs_assets",
    "hfs_sales",
    "cost_fin_assets",
    "fair_value_fin_assets",
    "contract_assets",
    "contract_liab",
    "accounts_receiv_bill",
    "accounts_pay",
    "oth_rcv_total",
    "fix_assets_total",
    "cip_total",
    "oth_pay_total",
    "long_pay_total",
    "debt_invest",
    "oth_debt_invest",
    "update_flag",
    "oth_eq_invest",
    "oth_illiq_fin_assets",
    "oth_eq_ppbond",
    "receiv_financing",
    "use_right_assets",
    "lease_liab"]
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
CREATE TABLE 'tsdata'.'ttsbalancesheet' (
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
df1 = tools.download_tushare(pro=pro, api_name='balancesheet_vip', fields=fields, limit=5000, start_date=start_date, report_type=1)
df4 = tools.download_tushare(pro=pro, api_name='balancesheet_vip', fields=fields, limit=5000, start_date=start_date, report_type=4)
df5 = tools.download_tushare(pro=pro, api_name='balancesheet_vip', fields=fields, limit=5000, start_date=start_date, report_type=5)
df11 = tools.download_tushare(pro=pro, api_name='balancesheet_vip', fields=fields, limit=5000, start_date=start_date, report_type=11)
df = pd.concat([df1,df4,df5,df11],axis=0)
if len(df) > 0:
    # df.loc[:, 'jyzc'] = df.loc[:, 'money_cap'].fillna(0) + df.loc[:, 'notes_receiv'].fillna(0) + df.loc[:, 'accounts_receiv'].fillna(0) + df.loc[:, 'prepayment'].fillna(0) + df.loc[:, 'inventories'].fillna(0) + df.loc[:, 'fix_assets_total'].fillna(0) + df.loc[:, 'produc_bio_assets'].fillna(0) + df.loc[:, 'oil_and_gas_assets'].fillna(0) + df.loc[:, 'intan_assets'].fillna(0) + df.loc[:, 'receiv_financing'].fillna(0) + df.loc[:, 'contract_assets'].fillna(0)
    # df.loc[:, 'jrzf'] = df.loc[:, 'lt_borr'].fillna(0) + df.loc[:, 'st_borr'].fillna(0) + df.loc[:, 'trading_fl'].fillna(0) + df.loc[:, 'sold_for_repur_fa'].fillna(0) + df.loc[:, 'int_payable'].fillna(0) + df.loc[:, 'div_payable'].fillna(0) + df.loc[:, 'st_bonds_payable'].fillna(0) + df.loc[:, 'non_cur_liab_due_1y'].fillna(0) + df.loc[:, 'bond_payable'].fillna(0) + df.loc[:, 'deriv_liab'].fillna(0) + df.loc[:, 'st_fin_payable'].fillna(0)
    df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]

    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

    df.to_sql('ttsbalancesheet', engine, schema='tsdata', index=False, chunksize=1000, if_exists='append', method=tools.mysql_replace_into)
