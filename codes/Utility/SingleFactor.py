
import datetime
import os
import sys
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tushare as ts
import tools
import Global_Config as gc
import pdb

class SingleFactor:
    def __init__(self, factor_name, stocks=None, start_date=None, end_date=None, data_dic=None):
        self.factor_name = factor_name
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        
        self.factor = None
        self.data_dic = data_dic
        
        
    def generate_factor(self, data=None):
        self.factor = None
        
    def inf_to_nan(self, factor):
        factor[factor==np.inf] = np.nan
        factor[factor==-np.inf] = np.nan
        return factor
    
    def factor_analysis(self, industry_neutral=True, neutral_risk=['MC', 'BP'], size_neutral=True, num_group=10):
        self.factor = self.inf_to_nan(self.factor)
        stocks = self.stocks
        start_date = self.start_date
        end_date = self.end_date
        y = pd.read_csv('%s/Data/y.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0])
        r_jiaoyi = pd.read_csv('%s/Data/r_jiaoyi.csv'%gc.LABELBASE_PATH, index_col=[0], parse_dates=[0])
        stocks_y = list(y.columns)
        stocks = list(set(stocks).intersection(stocks_y))
        y = y.loc[:, stocks]
        r_jiaoyi = r_jiaoyi.loc[:, stocks]
        
        if start_date:
            y = y.loc[y.index >= start_date, :]
            r_jiaoyi = r_jiaoyi.loc[r_jiaoyi.index >= start_date, :]
        
        if end_date:
            y = y.loc[y.index <= end_date, :]
            r_jiaoyi = r_jiaoyi.loc[r_jiaoyi.index <= end_date, :]
        
        ys = [r_jiaoyi.shift(-n) for n in range(1)]
        
        if not os.path.exists('%s/Results/%s'%(gc.SINGLEFACTOR_PATH, self.factor_name)):
            os.mkdir('%s/Results/%s'%(gc.SINGLEFACTOR_PATH, self.factor_name))
        self.factor = self.factor.loc[y.index, :]
        factor = self.factor.copy()
        #行业中性
        if industry_neutral:
            industrys = tools.get_industrys('L1', self.stocks)
            factor = tools.standardize_industry(self.factor, industrys)
            self.factor_industry_neutral = factor.copy()
        else:
            factor = tools.standardize(self.factor)
            self.factor_industry_neutral = None
        #市值中性
        #风险中性
        if neutral_risk:
            risk_dic = {risk:pd.read_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, risk), index_col=[0], parse_dates=[0]) for risk in neutral_risk}
            for risk in risk_dic.keys():
                risk_dic[risk] = tools.standardize_industry(risk_dic[risk], industrys)
            risk_df_list = [risk_df for risk_df in risk_dic.values()]
            def neutral_apply(y, x_list):
                date = y.name
                X = DataFrame(index=y.index)
                for x in x_list:
                    X = pd.concat([X, x.loc[date, :]], axis=1)
                X.fillna(0, inplace=True)
                # X = sm.add_constant(X)
                res = y - X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y.fillna(0)))
                return res
            factor = factor.apply(func=neutral_apply, args=(risk_df_list,), axis=1)
            self.factor_risk_neutral = factor.copy()
        else:
            self.factor_risk_neutral = None
        #因子分布
        plt.figure(figsize=(16,12))
        plt.hist(factor.fillna(0).values.flatten())
        plt.savefig('%s/Results/%s/hist.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))
        
        #IC、IR、分组回测
        IC = {}
        IR = {}
        
        for i in range(len(ys)):
            if industry_neutral:
                y_neutral = tools.standardize_industry(ys[i], industrys)
            else:
                y_neutral = tools.standardize(ys[i])
            IC[i] = (y_neutral * factor).mean(1) / factor.std(1) / y_neutral.std(1)
            IR[i] = IC[i].rolling(20).mean() / IC[i].rolling(20).std()
            
            factor_tmp = DataFrame(factor, index=ys[0].index, columns=ys[0].columns)
            factor_tmp[ys[0].isna()] = np.nan
            
            factor_quantile = DataFrame(factor_tmp.rank(axis=1), index=factor.index, columns=factor.columns).div(factor_tmp.notna().sum(1), axis=0)# / len(factor.columns)
            
            group_pos = {}
            for n in range(num_group):
                # if n == num_group - 1:
                #     pdb.set_trace()
                group_pos[n] = DataFrame((n/num_group <= factor_quantile) & (factor_quantile <= (n+1)/num_group))
                group_pos[n][~group_pos[n]] = np.nan
                group_pos[n] = 1 * group_pos[n]
            self.group_pos = group_pos
            
            plt.figure(figsize=(16, 12))
            group_mean = {}
            for n in range(num_group):
                group_mean[n] = ((group_pos[n] * ys[i]).mean(1) - 1*ys[i].mean(1)).cumsum().rename('%s'%(n/num_group))
                group_mean[n].plot()
            plt.legend(['%s'%i for i in range(num_group)])
            plt.savefig('%s/Results/%s/group_mean%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
            plt.figure(figsize=(16, 12))
            group_hist = [group_mean[i].iloc[np.where(group_mean[i].notna())[0][-1]] for i in range(num_group)]
            plt.bar(range(num_group), group_hist)
            plt.savefig('%s/Results/%s/group_mean_hist%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
            
            plt.figure(figsize=(16, 12))
            group_std = {}
            for n in range(num_group):
                group_std[n] = (group_pos[n] * ys[i]).std(1).cumsum().rename('%s'%(n/num_group))
                group_std[n].plot()
            plt.legend(['%s'%i for i in range(num_group)])
            plt.savefig('%s/Results/%s/group_std%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
            plt.figure(figsize=(16, 12))
            group_hist = [group_std[i].iloc[np.where(group_std[i].notna())[0][-1]] for i in range(num_group)]
            plt.bar(range(num_group), group_hist)
            plt.savefig('%s/Results/%s/group_std_hist%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
            
            plt.figure(figsize=(16, 12))
            group_skew = {}
            for n in range(num_group):
                group_skew[n] = (group_pos[n] * ys[i]).skew(1).cumsum().rename('%s'%(n/num_group))
                group_skew[n].plot()
            plt.legend(['%s'%i for i in range(num_group)])
            plt.savefig('%s/Results/%s/group_skew%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
            plt.figure(figsize=(16, 12))
            group_hist = [group_skew[i].iloc[np.where(group_skew[i].notna())[0][-1]] for i in range(num_group)]
            plt.bar(range(num_group), group_hist)
            plt.savefig('%s/Results/%s/group_skew_hist%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
            
            # plt.figure(figsize=(16, 12))
            # group_kurt = {}
            # for n in range(num_group):
            #     group_kurt[n] = (group_pos[n] * ys[i]).kurt(1).cumsum().rename('%s'%(n/num_group))
            #     group_kurt[n].plot()
            # plt.legend(['%s'%i for i in range(num_group)])
            # plt.savefig('%s/Results/%s/group_kurt%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
            # plt.figure(figsize=(16, 12))
            # group_hist = [group_kurt[i].iloc[np.where(group_kurt[i].notna())[0][-1]] for i in range(num_group)]
            # plt.bar(range(num_group), group_hist)
            # plt.savefig('%s/Results/%s/group_kurt_hist%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
        
        self.IC = IC
        self.IR = IR
        
        plt.figure(figsize=(16,12))
        for i in range(len(ys)):
            IC[i].cumsum().plot()
        plt.legend(['%s'%i for i in range(len(ys))])
        plt.savefig('%s/Results/%s/IC.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))
        
        plt.figure(figsize=(16,12))
        for i in range(len(ys)):
            IR[i].cumsum().plot()
        plt.legend(['%s'%i for i in range(len(ys))])
        plt.savefig('%s/Results/%s/IR.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))
        
        plt.figure(figsize=(16,12))
        for i in range(len(ys)):
            IC[i].abs().cumsum().plot()
        plt.legend(['%s'%i for i in range(len(ys))])
        plt.savefig('%s/Results/%s/IC_abs.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))
        
    def update_factor(self, data=None):
        self.generate_factor()
        if isinstance(self.factor, list):
            for i in range(len(self.factor)):
                factor = self.inf_to_nan(self.factor[i])
                if os.path.exists('%s/Data/%s%s.csv'%(gc.FACTORBASE_PATH, self.factor_name, self.n_list[i])):
                    factor_old = pd.read_csv('%s/Data/%s%s.csv'%(gc.FACTORBASE_PATH, self.factor_name, self.n_list[i]), index_col=[0], parse_dates=[0])
                
                    factor = pd.concat([factor_old.loc[factor_old.index<factor.index[0], :], factor], axis=0)
                factor.to_csv('%s/Data/%s%s.csv'%(gc.FACTORBASE_PATH, self.factor_name, self.n_list[i]))
            factor.iloc[-1,:].to_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name))
        else:
            factor = self.inf_to_nan(self.factor)
            if os.path.exists('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name)):
                factor_old = pd.read_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name), index_col=[0], parse_dates=[0])
                
                factor = pd.concat([factor_old.loc[factor_old.index<factor.index[0], :], factor], axis=0)
            factor.to_csv('%s/Data/%s.csv'%(gc.FACTORBASE_PATH, self.factor_name))