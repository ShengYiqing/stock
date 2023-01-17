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
    "comp_type",
    "report_type",
    "end_type",
    "net_profit",
    "finan_exp",
    "c_fr_sale_sg",
    "recp_tax_rends",
    "n_depos_incr_fi",
    "n_incr_loans_cb",
    "n_inc_borr_oth_fi",
    "prem_fr_orig_contr",
    "n_incr_insured_dep",
    "n_reinsur_prem",
    "n_incr_disp_tfa",
    "ifc_cash_incr",
    "n_incr_disp_faas",
    "n_incr_loans_oth_bank",
    "n_cap_incr_repur",
    "c_fr_oth_operate_a",
    "c_inf_fr_operate_a",
    "c_paid_goods_s",
    "c_paid_to_for_empl",
    "c_paid_for_taxes",
    "n_incr_clt_loan_adv",
    "n_incr_dep_cbob",
    "c_pay_claims_orig_inco",
    "pay_handling_chrg",
    "pay_comm_insur_plcy",
    "oth_cash_pay_oper_act",
    "st_cash_out_act",
    "n_cashflow_act",
    "oth_recp_ral_inv_act",
    "c_disp_withdrwl_invest",
    "c_recp_return_invest",
    "n_recp_disp_fiolta",
    "n_recp_disp_sobu",
    "stot_inflows_inv_act",
    "c_pay_acq_const_fiolta",
    "c_paid_invest",
    "n_disp_subs_oth_biz",
    "oth_pay_ral_inv_act",
    "n_incr_pledge_loan",
    "stot_out_inv_act",
    "n_cashflow_inv_act",
    "c_recp_borrow",
    "proc_issue_bonds",
    "oth_cash_recp_ral_fnc_act",
    "stot_cash_in_fnc_act",
    "free_cashflow",
    "c_prepay_amt_borr",
    "c_pay_dist_dpcp_int_exp",
    "incl_dvd_profit_paid_sc_ms",
    "oth_cashpay_ral_fnc_act",
    "stot_cashout_fnc_act",
    "n_cash_flows_fnc_act",
    "eff_fx_flu_cash",
    "n_incr_cash_cash_equ",
    "c_cash_equ_beg_period",
    "c_cash_equ_end_period",
    "c_recp_cap_contrib",
    "incl_cash_rec_saims",
    "uncon_invest_loss",
    "prov_depr_assets",
    "depr_fa_coga_dpba",
    "amort_intang_assets",
    "lt_amort_deferred_exp",
    "decr_deferred_exp",
    "incr_acc_exp",
    "loss_disp_fiolta",
    "loss_scr_fa",
    "loss_fv_chg",
    "invest_loss",
    "decr_def_inc_tax_assets",
    "incr_def_inc_tax_liab",
    "decr_inventories",
    "decr_oper_payable",
    "incr_oper_payable",
    "others",
    "im_net_cashflow_oper_act",
    "conv_debt_into_cap",
    "conv_copbonds_due_within_1y",
    "fa_fnc_leases",
    "im_n_incr_cash_equ",
    "net_dism_capital_add",
    "net_cash_rece_sec",
    "credit_impa_loss",
    "use_right_asset_dep",
    "oth_loss_asset",
    "end_bal_cash",
    "beg_bal_cash",
    "end_bal_cash_equ",
    "beg_bal_cash_equ",
    "update_flag"
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
CREATE TABLE 'tsdata'.'ttscashflow' (
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
df1 = tools.download_tushare(pro=pro, api_name='cashflow_vip', fields=fields, limit=5000, start_date=start_date, report_type=1)
df4 = tools.download_tushare(pro=pro, api_name='cashflow_vip', fields=fields, limit=5000, start_date=start_date, report_type=4)
df5 = tools.download_tushare(pro=pro, api_name='cashflow_vip', fields=fields, limit=5000, start_date=start_date, report_type=5)
df11 = tools.download_tushare(pro=pro, api_name='cashflow_vip', fields=fields, limit=5000, start_date=start_date, report_type=11)
df = pd.concat([df1,df4,df5,df11],axis=0)
if len(df) > 0:
    df.loc[:, 'stock_code'] = [sc.split('.')[0] for sc in df.loc[:, 'ts_code']]

    df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')

    df.to_sql('ttscashflow', engine, schema='tsdata', index=False, chunksize=1000, if_exists='append', method=tools.mysql_replace_into)
