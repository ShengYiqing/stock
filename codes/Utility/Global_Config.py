import os
import sys
import time
import datetime

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#字段名

#路径名
PROJECT_PATH = 'D:/stock'
GLOBALCONFIG_PATH = PROJECT_PATH + '/Codes'
DATABASE_PATH = PROJECT_PATH + '/DataBase'
FACTORBASE_PATH = PROJECT_PATH + '/FactorBase'
LABELBASE_PATH = PROJECT_PATH + '/LabelBase'
SINGLEFACTOR_PATH = PROJECT_PATH + '/SingleFactor'
MULTIFACTOR_PATH = PROJECT_PATH + '/MultiFactor'
MODEL_PATH = PROJECT_PATH + '/Model'
IC_PATH = PROJECT_PATH + '/IC'
BACKTEST_PATH = PROJECT_PATH + '/Backtest'
PREDICT_PATH = PROJECT_PATH + '/Predict'

#财报时间
END_DATE = ['0331', '0630', '0930', '1231']
ANN_DATE = ['0430', '0831', '1031', '0430']
UPD_DATE = ['0507', '0907', '1107', '0507']
DELTA_DATE = [50, 70, 40, 130]

#风险因子
WHITE_INDUSTRY_LIST = ['计算机', '商贸零售', '机械设备', '电力设备', '家用电器', 
                       '电子', '汽车', '医药生物', '传媒', '国防军工', 
                       '美容护理', '食品饮料']
# INDUSTRY_DIC = {
#     'iagri': ['农药化肥', '农业综合', '林业', '饲料', '种植业', '渔业', ], 
#     'ienergy': ['石油开采', '石油加工', '石油贸易', '煤炭开采',], 
#     'ibuildmat': ['普钢', '特种钢', '钢加工', '焦炭加工', '玻璃', '水泥', '其他建材', ], 
#     'ichemmat': ['化工原料',], 
#     'imine': ['矿物制品', '黄金', '铜', '铝', '铅锌', '小金属', ], 
#     'ichem': ['塑料', '橡胶', '化纤', '造纸', '染料涂料', ],
#     'ilight': ['纺织', '服饰', '家居用品', '陶瓷'], 
#     'iheavy': ['运输设备', '化工机械', '工程机械', '农用机械', '纺织机械', '轻工机械', '建筑工程', '装修装饰',], 
#     'istar': ['船舶', '航空', ], 
#     'iauto': ['汽车配件', '汽车整车', '摩托车', '汽车服务', ], 
#     'isoft': ['软件服务', '互联网', 'IT设备', '通信设备'], 
#     'ihard': ['元器件', '半导体'], 
#     'ielec': ['电气设备'], 
#     'itech': ['机床制造', '专用机械', '电器仪表', '机械基件', ], 
#     'ialco': ['白酒', '啤酒', '红黄酒', ], 
#     'idrinks': ['软饮料', '乳制品', ], 
#     'ifoods': ['食品'], 
#     'ibeauty': ['日用化工'], 
#     'icons': ['旅游景点', '酒店餐饮', '旅游服务', '影视音像', '文教休闲', '出版业', '家用电器', ], 
#     'imed': ['医药商业', '生物制药', '化学制药', '中成药', '医疗保健'], 
#     'ibusiness': ['其他商业', '商品城', '商贸代理', 
#                   '广告包装', '批发业', '仓储物流', 
#                   '百货', '超市连锁', '电器连锁', ], 
#     'ifin': ['银行', '证券', '保险', '多元金融', '全国地产', '区域地产', '房产服务', '园区开发'], 
#     'iutility': ['公共交通', '公路', '路桥', '铁路', '机场', '空运', '港口', '水运', 
#                  '环境保护', '供气供热', '水务', '电信运营', '火力发电', '水力发电', '新型电力', ], 
#     'iuk': ['综合类', None]
#     }

# INDUSTRY_DIC_INV = {}
# for ind_1 in INDUSTRY_DIC:
#     for ind_2 in INDUSTRY_DIC[ind_1]:
#         INDUSTRY_DIC_INV[ind_2] = ind_1

# WHITE_INDUSTRY_DIC = {
#     'iagri': [], 
#     'ienergy': [], 
#     'ibuildmat': [], 
#     'ichemmat': [], 
#     'imine': [], 
#     'ichem': [],
#     'ilight': [], 
#     'iheavy': ['运输设备', '化工机械', '工程机械', '农用机械', '纺织机械', '轻工机械', ], 
#     'istar': ['船舶', '航空', ], 
#     'iauto': ['汽车配件', '汽车整车', '摩托车', '汽车服务', ], 
#     'isoft': ['软件服务', '互联网', 'IT设备', '通信设备'], 
#     'ihard': ['元器件', '半导体'], 
#     'ielec': ['电气设备'], 
#     'itech': ['机床制造', '专用机械', '电器仪表', '机械基件', ],
#     'ialco': ['白酒', '啤酒', '红黄酒', ], 
#     'idrinks': ['软饮料', '乳制品', ], 
#     'ifoods': ['食品'], 
#     'ibeauty': ['日用化工'], 
#     'icons': ['旅游服务', '影视音像', '文教休闲', '家用电器', ], 
#     'imed': ['生物制药', '化学制药', '中成药', '医疗保健'], 
#     'ibusiness': ['广告包装', '仓储物流',], 
#     'ifin': [], 
#     'iutility': ['环境保护', '水力发电', '新型电力', ], 
#     'iuk': []
#     }

# WHITE_INDUSTRY_LIST = sum(list(WHITE_INDUSTRY_DIC.values()), [])

FACTORS = [
    'MC', 'BP',
    'Momentum', 'Sigma', 'CORRMarket', 'TurnRate', 'STR',
    'FJZCSYL', 'FJZCSYLDT',]