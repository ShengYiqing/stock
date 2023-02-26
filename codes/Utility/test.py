import os
import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import Global_Config as gc
import tools
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import tools

df = DataFrame(np.arange(80).reshape(20, 4))
df.rolling(10, win_type='exponential').mean()

trade_date = '20220104'
x1 = 'corrmarket'
x2 = 'hfcorrmarket'
def dailyhffactor_analysis(x1, x2, trade_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    
    sql = """
    select stock_code, factor_name , factor_value from intermediate.tdailyhffactor
    where trade_date = {trade_date}
    and factor_name in ('{x1}', '{x2}')
    """.format(x1=x1, x2=x2, trade_date=trade_date)
    df = pd.read_sql(sql, engine).set_index(['stock_code', 'factor_name']).unstack()
    df.dropna(inplace=True)
    plt.scatter(df.iloc[:, 0].rank(), df.iloc[:, 1].rank())
    print(df.corr())
dailyhffactor_analysis(x1, x2, trade_date)

for ind_k in gc.WHITE_INDUSTRY_DIC.keys():
    if (len(gc.WHITE_INDUSTRY_DIC[ind_k]) > 0):
        sql = """
        select count(1) from tsdata.ttsstockbasic where industry in %s
        """%gc.WHITE_INDUSTRY_DIC[ind_k]
        sql = sql.replace('[', '(').replace(']', ')')
        engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
        df = pd.read_sql(sql, engine)
        print(ind_k, df.iloc[0,0])
def quality_analyse(stock_code):
    files = os.listdir('D:/stock/DataBase/Data/factor')
    tables = [i.split('.')[0] for i in files]
    tables = list(filter(lambda x:x[7]=='f', tables))
    
    sql = """ select tfactorquality.trade_date, tfactorquality.stock_code, tfactorquality.preprocessed_factor_value quality """
    for table in tables:
        sql += ' , %s.preprocessed_factor_value %s'%(table, table[7:])
    sql += ' from tfactorquality '
    for table in tables:
        sql += ' left join %s on tfactorquality.trade_date = %s.trade_date and tfactorquality.stock_code = %s.stock_code '%(table, table, table)
    sql += """where tfactorquality.trade_date = '20230103' and tfactorquality.stock_code in %s"""%stock_code
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    df = pd.read_sql(sql, engine)
    return df.T
stock_code = '(002755)'
quality_analyse(stock_code)


files = os.listdir('D:/stock/DataBase/Data/factor')
tables = [i.split('.')[0] for i in files]
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
tables = ['tfactoramount']
for table in tables:
    print(table)
    sql = """optimize table factor.%s"""%table
    with engine.connect() as conn:
        conn.execute(sql)


trade_date = '20221212'
x1 = 'corrmarket'
x2 = 'hfcorrmarket'
tools.colinearity_analysis(x1, x2, trade_date)

start_date = '20200101'
end_date = '20221201'

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")

sql = """
select t1.trade_date as trade_date, t2.industry as industry, avg(r_avg) as r from label.tdailylabel t1
left join tsdata.ttsstockbasic t2
on t1.stock_code = t2.stock_code
where t1.trade_date >= {start_date}
and t1.trade_date <= {end_date}
and t2.industry not in ('农药化肥', '农业综合', '林业', '饲料', '种植业', '渔业', '农用机械', 
'石油开采', '石油加工', '石油贸易', '煤炭开采', '焦炭加工', 
'矿物制品', '普钢', '特种钢', '钢加工', 
'黄金', '铜', '铝', '铅锌', '小金属', 
'化工原料', '塑料', '橡胶', '化纤', '造纸', '染料涂料', 
'玻璃', '水泥', '其他建材', 
'银行', '证券', '保险', '多元金融', 
'全国地产', '区域地产', '房产服务', '园区开发', 
'环境保护', '建筑工程', '装修装饰', 
'公共交通', '公路', '路桥', '铁路', '机场', '空运', '港口', '水运', 
'供气供热', '水务', '电信运营', '火力发电', '水力发电', '新型电力', 
'综合类')
group by t1.trade_date, t2.industry
""".format(start_date=start_date, end_date=end_date)
df = pd.read_sql(sql, engine).set_index(['trade_date', 'industry']).loc[:, 'r'].unstack()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, n_init="auto").fit(df.T)
ks = kmeans.labels_
dic = {i:[] for i in range(10)}
for i in range(len(ks)):
    dic[ks[i]].append(df.columns[i])
dic
sql_ic = """
select trade_date, factor_name, (ic_avg+rank_ic_avg)/2 as ic from tdailyic
where factor_name in ('mc', 'bp', 'momentum', 'tr', 'corrmarket')
and field = 'white'
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_ic = sql_ic.format(start_date=start_date, end_date=end_date)
df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name']).loc[:, 'ic'].unstack()
df_ic.abs().cumsum().plot()

dic = {
    'ilight': ['纺织', '纺织机械', '轻工机械', '家用电器', '服饰', '家居用品', '陶瓷'], 
    'iheavy': ['运输设备', '航空', '化工机械', '船舶', '工程机械', '农用机械'], 
    'iauto': ['汽车配件', '汽车整车', '摩托车', '汽车服务', ], 
    'isoft': ['软件服务', '互联网', 'IT设备', '通信设备'], 
    'ihard': ['元器件', '半导体'], 
    'ielec': ['电气设备'], 
    'itech': ['机床制造', '专用机械', '电器仪表', '机械基件', ],
    'icons': ['食品', '白酒', '啤酒', '软饮料', '红黄酒', '乳制品', '日用化工', '旅游景点', '影视音像', '酒店餐饮', '旅游服务', '文教休闲', '出版业', ], 
    'imed': ['医药商业', '生物制药', '化学制药', '中成药', '医疗保健'], 
    'ibusiness': ['其他商业', '商品城', '商贸代理', '百货', '广告包装', '批发业', '超市连锁', '电器连锁', '仓储物流'], 
    }
l = []
for v in dic.values():
    l.extend(list(v))


l2 = [
    '农药化肥', '农业综合', '林业', '饲料', '种植业', '渔业', 
    '石油开采', '石油加工', '石油贸易', '煤炭开采', '焦炭加工', 
    '矿物制品', '普钢', '特种钢', '钢加工', 
    '黄金', '铜', '铝', '铅锌', '小金属', 
    '化工原料', '塑料', '橡胶', '化纤', '造纸', '染料涂料', 
    '玻璃', '水泥', '其他建材', 
    '银行', '证券', '保险', '多元金融', 
    '全国地产', '区域地产', '房产服务', '园区开发', 
    '环境保护', '建筑工程', '装修装饰', 
    '公共交通', '公路', '路桥', '铁路', '机场', '空运', '港口', '水运', 
    '供气供热', '水务', '电信运营', '火力发电', '水力发电', '新型电力', 
    '综合类']
len(l+l2)
len(set(l+l2))
