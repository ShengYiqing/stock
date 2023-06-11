import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import datetime
import time
from scipy.stats import rankdata
import tushare as ts
import Global_Config as gc
import statsmodels.api as sm
import multiprocessing as mp
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import pdb

def colinearity_analysis(x1, x2, trade_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    
    sql = """
    select t1.trade_date as trade_date,
     t1.stock_code as stock_code,
     t1.factor_value as {x1}, t2.factor_value as {x2}
    from factor.tfactor{x1} t1 left join
    factor.tfactor{x2} t2
    on t1.trade_date = t2.trade_date
    and t1.stock_code = t2.stock_code
    where t1.trade_date = {trade_date}""".format(x1=x1, x2=x2, trade_date=trade_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    df.dropna(inplace=True)
    plt.figure(figsize=(16, 9))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.figure(figsize=(16, 9))
    plt.scatter(df.iloc[:, 0].rank(), df.iloc[:, 1].rank())
    print(df.corr(method='spearman'))

def rolling_weight_sum(df_sum, df_weight, n, weight_type):
    columns = sorted(set(list(df_sum.columns)).intersection(set(list(df_weight.columns))))
    df_sum = DataFrame(df_sum, columns=columns)
    df_weight = DataFrame(df_weight, index=df_sum.index, columns=columns)
    df_return = DataFrame(index=df_sum.index, columns=columns)
    
    for i in range(n - 1, len(df_sum)):
        i_start = i - n + 1
        i_end = i
        if weight_type == 'rank':
            weight = df_weight.iloc[i_start:i_end, :].rank()
        if weight_type == 'central_rank':
            weight = df_weight.iloc[i_start:i_end, :].rank()
            weight = weight - weight.mean()
        if weight_type == 'median':
            weight = df_weight.iloc[i_start:i_end, :].copy()
            weight = weight - weight.median()
            weight[weight>0] = 1
            weight[weight<0] = 1
        
        df_return.iloc[i, :] = (df_sum.iloc[i_start:i_end, :] * weight).sum()

    return df_return


def factor_analyse(x, y, num_group, factor_name):
    #因子分布
    plt.figure(figsize=(16,12))
    plt.hist(x.values.flatten())
    plt.title(factor_name+'-hist')
    # plt.savefig('%s/Results/%s/hist.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))
    
    IC = x.corrwith(y, axis=1)
    IR = IC.rolling(20).mean() / IC.rolling(20).std()
    
    plt.figure(figsize=(16,12))
    IC.cumsum().plot()
    plt.title(factor_name+'-ic')
    # plt.savefig('%s/Results/%s/IC.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))
    
    plt.figure(figsize=(16,12))
    IR.cumsum().plot()
    plt.title(factor_name+'-ir')
    # plt.savefig('%s/Results/%s/IR.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))
    
    
    plt.figure(figsize=(16,12))
    (IC**2).cumsum().plot()
    plt.title(factor_name+'-r2')
    # plt.legend(['%s'%i for i in range(len(ys))])
    # plt.savefig('%s/Results/%s/IC_abs.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))
        
        
    x_quantile = DataFrame(x.rank(axis=1, pct=True))
    
    group_pos = {}
    for n in range(num_group):
        group_pos[n] = DataFrame((n/num_group <= x_quantile) & (x_quantile <= (n+1)/num_group))
        group_pos[n][~group_pos[n]] = np.nan
        group_pos[n] = 1 * group_pos[n]
        
    plt.figure(figsize=(16, 12))
    group_mean = {}
    for n in range(num_group):
        group_mean[n] = ((group_pos[n] * y).mean(1)+1).cumprod().rename('%s'%(n/num_group))
        group_mean[n].plot()
    plt.legend(['%s'%i for i in range(num_group)], loc="best")

    plt.figure(figsize=(16, 12))
    group_mean = {}
    for n in range(num_group):
        group_mean[n] = (group_pos[n] * y).mean(1).cumsum().rename('%s'%(n/num_group))
        group_mean[n].plot()
    plt.legend(['%s'%i for i in range(num_group)])
    #plt.title(factor_name+'-group backtest')
    # plt.savefig('%s/Results/%s/group_mean%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
    
    plt.figure(figsize=(16, 12))
    group_mean = {}
    for n in range(num_group):
        group_mean[n] = ((group_pos[n] * y).mean(1) - 1*y.mean(1)).cumsum().rename('%s'%(n/num_group))
        group_mean[n].plot()
    plt.legend(['%s'%i for i in range(num_group)])
    plt.title(factor_name+'-group alpha')
    # plt.savefig('%s/Results/%s/group_mean%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
    
    plt.figure(figsize=(16, 12))
    group_hist = [group_mean[i].iloc[np.where(group_mean[i].notna())[0][-1]] for i in range(num_group)]
    plt.bar(range(num_group), group_hist)
    plt.title(factor_name+'-group alpha')
    # plt.savefig('%s/Results/%s/group_mean_hist%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
    
    plt.figure(figsize=(16, 12))
    group_std = {}
    for n in range(num_group):
        group_std[n] = (group_pos[n] * y).std(1).cumsum().rename('%s'%(n/num_group))
        group_std[n].plot()
    plt.legend(['%s'%i for i in range(num_group)])
    plt.title(factor_name+'-group return std')
    # plt.savefig('%s/Results/%s/group_std%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
    
    plt.figure(figsize=(16, 12))
    group_hist = [group_std[i].iloc[np.where(group_std[i].notna())[0][-1]] for i in range(num_group)]
    plt.bar(range(num_group), group_hist)
    plt.title(factor_name+'-group return std')
    # plt.savefig('%s/Results/%s/group_std_hist%s.png'%(gc.SINGLEFACTOR_PATH, self.factor_name, i))
    
    
def generate_sql_y_x(factor_names, start_date, end_date, is_trade=True, is_white=True, is_industry=False):
    sql = ' select t1.trade_date, t1.stock_code, t1.r_daily, t1.r_weekly, t1.r_monthly '
    
    for factor_name in factor_names:
        sql += ' , t{factor_name}.factor_value {factor_name} '.format(factor_name=factor_name)
    sql += ' from label.tdailylabel t1 '
    for factor_name in factor_names:
        sql += """ left join factor.tfactor{factor_name} t{factor_name} 
                   on t1.trade_date = t{factor_name}.trade_date 
                   and t1.stock_code = t{factor_name}.stock_code """.format(factor_name=factor_name)
    
    if is_industry:
        sql += """ left join indsw.tindsw t3
                   on t1.stock_code = t3.stock_code """
    sql += """ where t1.trade_date >= \'{start_date}\'
               and t1.trade_date <= \'{end_date}\'""".format(start_date=start_date, end_date=end_date)
    if is_trade:
        sql += " and t1.is_trade = 1 "
    if is_white:
        sql += " and t1.is_white = 1 "
    if is_industry:
        sql += (" and t3.l3_name in %s "%gc.WHITE_INDUSTRY_LIST).replace('[', '(').replace(']', ')')
    return sql


def trade_date_shift(date, shift):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    sql = """
    select distinct(cal_date) from ttstradecal
    where is_open = 1
    """
    trade_cal = pd.read_sql(sql, engine).loc[:, 'cal_date']
    n = np.where(trade_cal<=date)[0][-1] - shift + 1
    if n < 0:
        n = 0
    return trade_cal.loc[n]


def download_tushare(pro, api_name, limit=None, retry=10, pause=30, **kwargs):
    for _ in range(retry):
        try:
            if limit:
                df = DataFrame()
                i = 1
                n = 1
                offset = 0
                while n > 0:
                    print(i)
                    df_tmp = pro.query(api_name=api_name, limit=limit, offset=offset, **kwargs)
                    n = len(df_tmp)
                    df = pd.concat([df, df_tmp], ignore_index=True)
                    offset = offset + n
                    i = i + 1
            else:
                df = pro.query(api_name=api_name, **kwargs)
        except:
            print('fail')
            time.sleep(pause)
        else:
            return df
    return DataFrame()
       
 
def mysql_replace_into(table, conn, keys, data_iter):
    from sqlalchemy.dialects.mysql import insert

    data = [dict(zip(keys, row)) for row in data_iter]

    stmt = insert(table.table).values(data)
    update_stmt = stmt.on_duplicate_key_update(**dict(zip(stmt.inserted.keys(), 
                                               stmt.inserted.values())))

    conn.execute(update_stmt)


def get_trade_cal(start_date, end_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    sql_trade_cal = """
    select distinct cal_date from ttstradecal where is_open = 1
    """
    
    trade_cal = list(pd.read_sql(sql_trade_cal, engine).loc[:, 'cal_date'])
    trade_cal = list(filter(lambda x:(x>=start_date) & (x<=end_date), trade_cal))
    return trade_cal


def reg_ts(df, n):
    x = np.arange(n)
    x = x - x.mean()
    b = df.rolling(n).apply(lambda y:(y*x).sum() / (x*x).sum(), raw=True)
    a = df.rolling(n).mean()
    y_hat = a + b * x[-1]
    e = df - y_hat
    
    return b, e


def neutralize(data, factors=['mc', 'bp', 'reversal', 'tr'], ind=None):
    if isinstance(data, DataFrame):
        data.index.name = 'trade_date'
        data.columns.name = 'stock_code'
        engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
        f0 = factors[0]
        sql = """
        select t{f0}.trade_date trade_date, t{f0}.stock_code stock_code, t{f0}.factor_value {f0}
        """.format(f0=f0)
        if len(factors) > 1:
            for f in factors[1:]:
                sql += """
                , t{f}.factor_value {f}
                """.format(f=f)
        sql += """
        , tind.l1_name l1, tind.l2_name l2, tind.l3_name l3
        """
        
        sql += """
        from factor.tfactor{f0} t{f0}
        """.format(f0=f0)
        if len(factors) > 1:
            for f in factors[1:]:
                sql += """
                left join factor.tfactor{f} t{f}
                on t{f0}.stock_code = t{f}.stock_code
                and t{f0}.trade_date = t{f}.trade_date
                """.format(f0=f0, f=f)
        
        if len(data.index) > 1:
            sql += """
            left join indsw.tindsw tind
            on t{f0}.stock_code = tind.stock_code
            where t{f0}.trade_date in {trade_dates}
            and t{f0}.stock_code in {stock_codes}
            """.format(f0=f0, trade_dates=tuple(data.index), stock_codes=tuple(data.columns))
        else:
            sql += """
            left join indsw.tindsw tind
            on t{f0}.stock_code = tind.stock_code
            where t{f0}.trade_date = {trade_date}
            and t{f0}.stock_code in {stock_codes}
            """.format(f0=f0, trade_date=data.index[0], stock_codes=tuple(data.columns))
        df_n = pd.read_sql(sql, engine)
        df_n = df_n.set_index(['trade_date', 'stock_code'])
        y = data.stack()
        y.name = 'y'
        data = pd.concat([y, df_n], axis=1).dropna()

        def g(data):
            # pdb.set_trace()
            if ind == None:
                X = data.loc[:, factors].fillna(0)
            else:
                X = pd.concat([pd.get_dummies(data.loc[:, ind]), data.loc[:, factors]], axis=1).fillna(0)
            
            X = sm.add_constant(X)
            for factor in factors:
                X.loc[:, factor] = winsorize(X.loc[:, factor])
            
            y = winsorize(data.loc[:, 'y'])
            
            y_predict = X.dot(np.linalg.inv(X.T.dot(X)+0.001*np.identity(len(X.T))).dot(X.T).dot(y))
            res = standardize(winsorize(y - y_predict))
            return res
        x_n = data.groupby('trade_date', as_index=False).apply(g).reset_index(0, drop=True)

        return x_n.unstack()
    else:
        return None
    
def centralize(data):
    return data.subtract(data.mean(1), 0)

def standardize(data):
    if isinstance(data, DataFrame):
        if len(data.columns) > 1:
            if (data.std(1) == 0).any():
                return data.subtract(data.mean(1), 0)
            else:
                return data.subtract(data.mean(1), 0).divide(data.std(1), 0)
        else:
            return data.subtract(data.mean(1), 0)
    elif isinstance(data, Series):
        return (data - data.mean()) / data.std()
    else:
        return None
    
def truncate(df, percent=0.025):
    tmp = df.copy()
    q1 = tmp.quantile(percent, 1)
    q2 = tmp.quantile(1-percent, 1)
    tmp[tmp.le(q1, 0)] = np.nan
    tmp[tmp.ge(q2, 0)] = np.nan
    
    return tmp

def winsorize(df, percent=0.025):
    tmp = df.copy()
    if isinstance(df, DataFrame):
        def f(s):
            q1 = s.quantile(percent)
            q2 = s.quantile(1-percent)
            s[s<q1] = q1
            s[s>q2] = q2
            return s
        tmp = tmp.apply(func=f, axis=1, result_type='expand')
        
        return tmp
    elif isinstance(df, Series):
        q1 = tmp.quantile(percent)
        q2 = tmp.quantile(1-percent)
        tmp[tmp<q1] = q1
        tmp[tmp>q2] = q2
        
        return tmp
    else:
        return None