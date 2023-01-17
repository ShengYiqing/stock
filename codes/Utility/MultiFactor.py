
import datetime
import os
import sys
import pickle
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import tools
import Global_Config as gc
import seaborn as sns

class MultiFactor:
    def __init__(self, factor_name, stocks, start_date=None, end_date=None):
        self.factor_name = factor_name
        self.start_date = start_date
        self.end_date = end_date
        self.stocks = stocks
        self.factor = None
        self.factor_list = None
        self.method = None
        self.quantile_nl = None
        
    
    def set_factor(self):
        self.factor_list = None
        self.method = None
        self.quantile_nl = None
        
    def get_factor(self):
        if self.factor_list == 'all':
            files = os.listdir('%s/Data/'%(gc.FACTORBASE_PATH))
            self.factor_dict = {file.split('.')[0]:pd.read_csv('%s/Data/%s'%(gc.FACTORBASE_PATH, file), index_col=[0], parse_dates=[0]) for file in files}
        else:
            self.factor_dict = {factor:pd.read_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, factor), index_col=[0], parse_dates=[0]) for factor in self.factor_list}
        for key in self.factor_dict.keys():
            self.factor_dict[key] = self.factor_dict[key].loc[(self.factor_dict[key].index>self.start_date) & (self.factor_dict[key].index<self.end_date), :]
        self.df = DataFrame({factor:self.factor_dict[factor].values.reshape(-1) for factor in self.factor_list})
        self.corr = self.df.corr()
        self.e_value, self.e_vector = np.linalg.eig(self.corr)
        r = np.array(len(self.e_value) - rankdata(self.e_value), dtype=np.int32)
        self.e_value = self.e_value[r]
        self.e_vector = self.e_vector[:, r]
    
    def pairplot(self):
        plt.figure(figsize=(16,12))
        sns.pairplot(self.df)
        plt.savefig('%s/Results/%s/pair.png'%(gc.MULTIFACTOR_PATH, self.factor_name))
        
    def corrplot(self):
        plt.figure(figsize=(16,12))
        sns.heatmap(self.corr)
        plt.savefig('%s/Results/%s/corr.png'%(gc.MULTIFACTOR_PATH, self.factor_name))
    
    def screeplot(self):
        plt.figure(figsize=(16,12))
        plt.plot(self.e_value / self.e_value.sum())
        plt.savefig('%s/Results/%s/scree.png'%(gc.MULTIFACTOR_PATH, self.factor_name))
    
    def multi_analysis(self):
        if not os.path.exists('%s/Results/%s'%(gc.MULTIFACTOR_PATH, self.factor_name)):
            os.mkdir('%s/Results/%s'%(gc.MULTIFACTOR_PATH, self.factor_name))
        self.pairplot()
        self.corrplot()
        self.screeplot()
        
    def combine_factor(self):
        self.factor = DataFrame()
        if self.method == 'ew':
            for factor in self.factor_list:
                self.factor = self.factor.add(self.factor_dict[factor], fill_value=0)
        elif self.method[:4] == 'pca_':
            pca_num = int(self.method[4])
            for i in range(len(self.factor_list)):
                self.factor = self.factor.add(self.e_vector[i, pca_num] * self.factor_dict[self.factor_list[i]], fill_value=0)
        if self.quantile_nl:
            self.factor = self.factor.subtract(self.factor.quantile(self.quantile_nl, axis=1), axis=0) ** 2
    def inf_to_nan(self, factor):
        factor[factor==np.inf] = np.nan
        factor[factor==-np.inf] = np.nan
        return factor
    
    def factor_analysis(self, industry_neutral=True, size_neutral=True, num_group=10):
        self.factor = self.inf_to_nan(self.factor)
        stocks = self.stocks
        start_date = self.start_date
        end_date = self.end_date
        y1 = pd.read_csv('%s/Data/y1.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0]).loc[:, stocks]
        y2 = pd.read_csv('%s/Data/y2.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0]).loc[:, stocks]
        y3 = pd.read_csv('%s/Data/y3.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0]).loc[:, stocks]
        y4 = pd.read_csv('%s/Data/y4.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0]).loc[:, stocks]
        y5 = pd.read_csv('%s/Data/y5.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0]).loc[:, stocks]
        
        if start_date:
            y1 = y1.loc[y1.index >= start_date, :]
            y2 = y2.loc[y2.index >= start_date, :]
            y3 = y3.loc[y3.index >= start_date, :]
            y4 = y4.loc[y4.index >= start_date, :]
            y5 = y5.loc[y5.index >= start_date, :]
            
        if end_date:
            y1 = y1.loc[y1.index <= end_date, :]
            y2 = y2.loc[y2.index <= end_date, :]
            y3 = y3.loc[y3.index <= end_date, :]
            y4 = y4.loc[y4.index <= end_date, :]
            y5 = y5.loc[y5.index <= end_date, :]
            
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.y5 = y5
        
        if not os.path.exists('%s/Results/%s/%s'%(gc.MULTIFACTOR_PATH, self.factor_name, self.method)):
            os.mkdir('%s/Results/%s/%s'%(gc.MULTIFACTOR_PATH, self.factor_name, self.method))
        factor = self.factor.copy()
        
        #行业中性
        if industry_neutral:
            industrys = tools.get_industrys('L1', self.stocks)
            tmp = {}
            for k in industrys.keys():
                if len(industrys[k]) > 0:
                    tmp[k] = industrys[k]
            industrys = tmp
            factor = tools.standardize_industry(self.factor, industrys)
            self.factor_industry_neutral = factor.copy()
        
        #市值中性
        if size_neutral:
            market_capitalization = DataFrame({stock: pd.read_csv('%s/StockTradingDerivativeData/Stock/%s.csv'%(gc.DATABASE_PATH, stock), index_col=[0], parse_dates=[0]).loc[:, 'TOTMKTCAP'] for stock in self.stocks})
            market_capitalization = np.log(market_capitalization)
            if self.start_date:
                market_capitalization = market_capitalization.loc[market_capitalization.index >= self.start_date, :]
            if self.end_date:
                market_capitalization = market_capitalization.loc[market_capitalization.index <= self.end_date, :]
            if industry_neutral:
                market_capitalization = tools.standardize_industry(market_capitalization, industrys)
            beta = (factor * market_capitalization).sum(1) / (market_capitalization * market_capitalization).sum(1)
            factor = factor - market_capitalization.mul(beta, axis=0)
            self.factor_industry_size_neutral = factor.copy()
        
        # self.factor_industry_neutral.fillna(0, inplace=True)
        # self.factor_industry_size_neutral.fillna(0, inplace=True)
        # factor.fillna(0, inplace=True)
        #因子分布
        plt.figure(figsize=(16,12))
        plt.hist(factor.fillna(0).values.flatten())
        plt.savefig('%s/Results/%s/%s/hist.png'%(gc.MULTIFACTOR_PATH, self.factor_name, self.method))
        
        #IC、IR、分组回测
        ys = [self.y1, self.y2, self.y3, self.y4, self.y5]
        IC = {}
        IR = {}
        group_backtest = {}
        group_pos = {}
        
        for i in range(len(ys)):
            if industry_neutral:
                y_neutral = tools.standardize_industry(ys[i], industrys)
            if size_neutral:
                y_neutral = y_neutral - market_capitalization.mul((y_neutral * market_capitalization).sum(1) / (market_capitalization * market_capitalization).sum(1), axis=0)
            IC[i] = (y_neutral * factor).mean(1) / factor.std(1) / y_neutral.std(1)
            IR[i] = IC[i].rolling(20).mean() / IC[i].rolling(20).std()
            factor_quantile = DataFrame(rankdata(factor, axis=1), index=factor.index, columns=factor.columns).div(factor.notna().sum(1), axis=0)# / len(factor.columns)
            factor_quantile[factor.isna()] = np.nan
            group_backtest[i] = {}
            group_pos[i] = {}
            for n in range(num_group):
                group_pos[i][n] = DataFrame((n/num_group <= factor_quantile) & (factor_quantile <= (n+1)/num_group))
                group_pos[i][n][~group_pos[i][n]] = np.nan
                group_pos[i][n] = 1 * group_pos[i][n]
                group_backtest[i][n] = ((group_pos[i][n] * ys[i]).mean(1) - ys[i].mean(1)).cumsum().rename('%s'%(n/num_group))
        self.IC = IC
        self.IR = IR
        self.group_pos = group_pos
        self.group_backtest = group_backtest
        
        plt.figure(figsize=(16,12))
        for i in range(len(ys)):
            IC[i].cumsum().plot()
        plt.legend(['%s'%i for i in range(len(ys))])
        plt.savefig('%s/Results/%s/%s/IC.png'%(gc.MULTIFACTOR_PATH, self.factor_name, self.method))
        
        plt.figure(figsize=(16,12))
        for i in range(len(ys)):
            IR[i].cumsum().plot()
        plt.legend(['%s'%i for i in range(len(ys))])
        plt.savefig('%s/Results/%s/%s/IR.png'%(gc.MULTIFACTOR_PATH, self.factor_name, self.method))
        
        for i in range(len(ys)):
            plt.figure(figsize=(16,12))
            for n in range(num_group):
                group_backtest[i][n].plot()
            plt.legend(['%s'%i for i in range(num_group)])
            plt.savefig('%s/Results/%s/%s/groupbacktest%s.png'%(gc.MULTIFACTOR_PATH, self.factor_name, self.method, i))
        
            
    def update_factor(self):
        self.set_factor()
        self.get_factor()
        self.combine_factor()
        self.factor = self.inf_to_nan(self.factor)
        #if 'industry' in self.neutral_list:
        if True:
            industrys = tools.get_industrys('L1', self.stocks)
            tmp = {}
            for k in industrys.keys():
                if len(industrys[k]) > 0:
                    tmp[k] = industrys[k]
            industrys = tmp
            factor = tools.standardize_industry(self.factor, industrys)
        #if 'market_capitalization' in self.neutral_list:
        if True:
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
        # factor.fillna(0, inplace=True)
        if os.path.exists('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name)):
            if isinstance(factor.index[0], str):
                factor_old = pd.read_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name), index_col=[0])
            else:
                factor_old = pd.read_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name), index_col=[0], parse_dates=[0])
            factor = pd.concat([factor_old, factor.loc[factor.index>factor_old.index[-1], :]], axis=0)
            factor.sort_index(axis=0, inplace=True)
        factor.sort_index(axis=1, inplace=True)
        factor.to_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name))