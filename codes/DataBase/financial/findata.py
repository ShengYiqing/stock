import os
import sys
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import Global_Config as gc
import tools
import sqlalchemy as sa
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")

end_date = datetime.datetime.today().strftime('%Y%m%d')

start_date = (datetime.datetime.today() - datetime.timedelta(30)).strftime('%Y%m%d')
# end_date = '20150101'
# start_date = '20220101'

# start_y = 2010
# end_y = 2023
# for i in range(2010, 2024):
#     start_date = str(i) + '0101'
#     end_date = str(i+1) + '0101'
#     print(start_date, end_date)
        
sql_bs = """
select 
ann_date, 
end_date, 
stock_code, 
total_hldr_eqy_exc_min_int jzc,
 
total_assets zzc, 
money_cap hbzj, 
notes_receiv yspj, 
accounts_receiv yszk, 
prepayment yfkx, 
receiv_financing yskxrz, 
contract_assets htzc, 
inventories ch, 
fix_assets_total gdzc, 
produc_bio_assets scxswzc, 
oil_and_gas_assets yqzc, 
intan_assets wxzc,
lt_borr cqjk, 
st_borr dqjk, 
trading_fl jyxjrfz, 
sold_for_repur_fa mchgjrzck, 
int_payable yflx, 
div_payable yfgl, 
st_bonds_payable yfdqzq, 
non_cur_liab_due_1y ynndqdfldfz, 
bond_payable yfzq, 
deriv_liab ysjrfz, 
st_fin_payable yfdqrzk 
from ttsbalancesheet
where ann_date >= {start_date} 
and ann_date <= {end_date}
""".format(start_date=start_date, end_date=end_date)
df_bs = pd.read_sql(sql_bs, engine)

sql_in = """
select 
ann_date, 
end_date, 
stock_code, 
revenue yysr, 
oper_cost yycb, 
biz_tax_surchg sjjfj, 
sell_exp xsfy, 
admin_exp glfy, 
fin_exp cwfy,  
compr_inc_attr_p gmjlr
from ttsincome 
where ann_date >= {start_date} 
and ann_date <= {end_date}
""".format(start_date=start_date, end_date=end_date)
df_in = pd.read_sql(sql_in, engine)

sql_cf = """
select 
ann_date, 
end_date, 
stock_code, 
n_cashflow_act jyxjll, 
stot_out_inv_act tzxjlc, 
stot_cash_in_fnc_act czxjlr
from ttscashflow
where ann_date >= {start_date} 
and ann_date <= {end_date}
""".format(start_date=start_date, end_date=end_date)
df_cf = pd.read_sql(sql_cf, engine)

df = df_bs.merge(df_in, on=['ann_date', 'end_date', 'stock_code'], how='outer').merge(df_cf, on=['ann_date', 'end_date', 'stock_code'], how='outer')
df.loc[:, 'ann_type'] = '定期报告'

sql_yg = """
select
ann_date, 
end_date, 
stock_code, 
(net_profit_min+net_profit_max)/2 * 10000 gmjlr
from ttsforecast
where ann_date >= {start_date} 
and ann_date <= {end_date}
""".format(start_date=start_date, end_date=end_date)
df_yg = pd.read_sql(sql_yg, engine)
df_yg.loc[:, 'ann_type'] = '业绩预告'

sql_kb = """
select
ann_date, 
end_date, 
stock_code, 
revenue yysr, 
n_income gmjlr, 
total_assets zzc, 
total_hldr_eqy_exc_min_int jzc
from ttsexpress
where ann_date >= {start_date} 
and ann_date <= {end_date}
""".format(start_date=start_date, end_date=end_date)
df_kb = pd.read_sql(sql_kb, engine)
df_kb.loc[:, 'ann_type'] = '业绩快报'

sql_yq = """
select
report_date ann_date, 
quarter end_date, 
stock_code, 
op_rt * 10000 yysr, 
np * 10000 gmjlr, 
eps, 
pe, 
rd, 
roe, 
ev_ebitda
from ttsreportrc
where report_date >= {start_date} 
and report_date <= {end_date}
and substr(quarter, -2) = 'Q4'
""".format(start_date=start_date, end_date=end_date)
df_yq = pd.read_sql(sql_yq, engine)
df_yq.loc[:, 'end_date'] = [i[:4]+'1231' for i in df_yq.loc[:, 'end_date']]
df_yq = df_yq.groupby(['ann_date', 'end_date', 'stock_code']).median().reset_index()
df_yq.loc[:, 'ann_type'] = '分析师预期'

df = pd.concat([df, df_yg, df_kb, df_yq])

df.loc[:, 'REC_CREATE_TIME'] = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
df = DataFrame(df.set_index(['ann_date', 'end_date', 'stock_code','ann_type', 'REC_CREATE_TIME']).stack())
df.index.names = ['ann_date', 'end_date', 'stock_code', 'ann_type', 'REC_CREATE_TIME', 'financial_index']
df.columns = ['financial_value']
df.to_sql('tfindata', engine, schema='findata', chunksize=10000, index=True, if_exists='append', method=tools.mysql_replace_into)