#!/usr/bin/env python
# coding: utf-8

#%%
import sys
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import os
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
from SingleFactor import SingleFactor
import Global_Config as gc
import tools
#%%

class HFVolPower(SingleFactor):
    def generate_factor(self):
        
        trade_cal = tools.get_trade_cal(self.start_date, self.end_date)
        power = DataFrame()
        
        def estimate_power(s, n=10):
            y = np.log(np.arange(1,n)/n)
            x = np.log(np.array([s.quantile(i) for i in np.arange(1,n)/n]))
            y = y - y.mean()
            x = x - x.mean()
            power = (x * y).sum() / (x * x).sum()
            return power
        
        for date in trade_cal:
            files = os.listdir('%s/StockSnapshootData/%s'%(gc.DATABASE_PATH, date))
            
            data_dic = {file.split('.')[1]:pd.read_csv('%s/StockSnapshootData/%s/%s'%(gc.DATABASE_PATH, date, file), index_col=[0], parse_dates=[0]) for file in files}
            keys = list(data_dic.keys())
            for key in keys:
                if len(data_dic[key]) == 0:
                    del data_dic[key]
            
            vol = DataFrame({'%s.SZ'%stock:data_dic[stock].loc[:, 'last_volume'] for stock in data_dic.keys()})
            vol.fillna(0, inplace=True)
            vol = vol.loc[vol.index>='%s093000'%(date.replace('-', '')), :]
            vol = vol.resample('1T').sum()
            vol[vol==0] = 1
            
            power_daily = vol.apply(func=estimate_power, axis=0)
            
            power = pd.concat([power, DataFrame({date:power_daily}).T], axis=0)

        a = power
        self.factor = a


    def update_factor(self):
        self.generate_factor()
        factor = self.factor
        #if 'industry' in self.neutral_list:
        if False:
            industrys = tools.get_industrys('L1', self.stocks)
            tmp = {}
            for k in industrys.keys():
                if len(industrys[k]) > 0:
                    tmp[k] = industrys[k]
            industrys = tmp
            factor = tools.standardize_industry(self.factor, industrys)
        #if 'market_capitalization' in self.neutral_list:
        if False:
            market_capitalization = DataFrame({stock: pd.read_csv('%s/StockTradingDerivativeData/Stock/%s.csv'%(gc.DATABASE_PATH, stock), index_col=[0], parse_dates=[0]).loc[:, 'TOTMKTCAP'] for stock in self.stocks})
            market_capitalization = np.log(market_capitalization)
            if self.start_date:
                market_capitalization = market_capitalization.loc[market_capitalization.index >= self.start_date, :]
            if self.end_date:
                market_capitalization = market_capitalization.loc[market_capitalization.index <= self.end_date, :]
            #if 'industry' in self.neutral_list:
            if True:
                market_capitalization = tools.standardize_industry(market_capitalization, industrys)
            beta = (factor * market_capitalization).sum(1) / (market_capitalization * market_capitalization).sum(1)
            factor = factor - market_capitalization.mul(beta, axis=0)
        #self.factor.fillna(0, inplace=True)
        if os.path.exists('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name)):
            # if isinstance(factor.index[0], str):
            #     factor_old = pd.read_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name), index_col=[0])
            #     factor_old.index = [str(i) for i in factor_old.index]
            # else:
            #     factor_old = pd.read_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name), index_col=[0], parse_dates=[0])
            factor_old = pd.read_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name), index_col=[0], parse_dates=[0])
            
            factor = pd.concat([factor_old.loc[factor_old.index<factor.index[0], :], factor], axis=0)
            #factor.sort_index(axis=0, inplace=True)
        factor.sort_index(axis=1, inplace=True)
        factor.to_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name))
#%%
if __name__ == '__main__':
    #????????????
    stocks = tools.get_stocks()

    a = HFVolPower('HFVolPower', stocks=stocks, start_date='20200901', end_date='20210128')
    
    a.generate_factor()
    
    a.factor_analysis()
    
    
