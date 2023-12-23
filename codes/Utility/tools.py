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

def colinearity_analysis(x1, x2, start_date, end_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
    
    sql = """
    select t1.trade_date as trade_date,
     t1.stock_code as stock_code,
     t1.factor_value as {x1}, t2.factor_value as {x2}
    from factor.tfactor{x1} t1 left join
    factor.tfactor{x2} t2
    on t1.trade_date = t2.trade_date
    and t1.stock_code = t2.stock_code
    where t1.trade_date >= {start_date}
    and t1.trade_date <= {end_date}
    """.format(x1=x1, x2=x2, start_date=start_date, end_date=end_date)
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    df.dropna(inplace=True)
    c = df.groupby('trade_date').apply(lambda x:x.corr(method='spearman').loc[x1, x2])
    plt.figure(figsize=(16, 9))
    (c**2).cumsum().plot()
    return c

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
    try:
        os.mkdir('%s/Factor/%s'%(gc.OUTPUT_PATH, factor_name))
    except:
        pass
    plt.figure(figsize=(16,9))
    plt.hist(x.values.flatten())
    plt.title(factor_name+'-hist')
    plt.savefig('%s/Factor/%s/00hist.png'%(gc.OUTPUT_PATH, factor_name))
    
    IC = x.corrwith(y, axis=1, method='spearman')
    IR = IC.rolling(250).mean() / IC.rolling(250).std()
    
    
    
    plt.figure(figsize=(16,9))
    IC.cumsum().plot()
    plt.title(factor_name+'-ic')
    plt.savefig('%s/Factor/%s/02ic.png'%(gc.OUTPUT_PATH, factor_name))
    
    plt.figure(figsize=(16,9))
    IR.cumsum().plot()
    plt.title(factor_name+'-ir')
    plt.savefig('%s/Factor/%s/03ir.png'%(gc.OUTPUT_PATH, factor_name))
    
    
    plt.figure(figsize=(16,9))
    (IC**2).cumsum().plot()
    plt.title(factor_name+'-r2')
    plt.savefig('%s/Factor/%s/04r2.png'%(gc.OUTPUT_PATH, factor_name))
        
        
    x_quantile = x.rank(axis=1, pct=True)
    
    group_pos = {}
    for n in range(num_group):
        group_pos[n] = DataFrame((n/num_group <= x_quantile) & (x_quantile <= (n+1)/num_group)).astype(int)
        # group_pos[n] = group_pos[n].rolling(n_hold).mean()
        group_pos[n] = group_pos[n].replace(0, np.nan)
        
    group_mean = {}
    for n in range(num_group):
        group_mean[n] = ((group_pos[n] * y).mean(1)+1).cumprod().rename('%s'%(n/num_group))
    DataFrame(group_mean).plot(figsize=(16, 9), cmap='coolwarm', title=factor_name)
    
    group_mean = {}
    for n in range(num_group):
        group_mean[n] = (group_pos[n] * y).mean(1).cumsum().rename('%s'%(n/num_group))
    DataFrame(group_mean).plot(figsize=(16, 9), cmap='coolwarm', title=factor_name)
    
    
    group_mean = {}
    for n in range(num_group):
        group_mean[n] = ((group_pos[n] * y).mean(1) - 1*y.mean(1)).cumsum().rename('%s'%(n/num_group))
    DataFrame(group_mean).plot(figsize=(16, 9), cmap='coolwarm', title=factor_name)
    
    plt.figure(figsize=(16, 9))
    group_hist = [group_mean[i].iloc[np.where(group_mean[i].notna())[0][-1]] for i in range(num_group)]
    plt.bar(range(num_group), group_hist)
    plt.title(factor_name)
    
    group_std = {}
    for n in range(num_group):
        group_std[n] = (group_pos[n] * y).std(1).cumsum().rename('%s'%(n/num_group))
    DataFrame(group_std).plot(figsize=(16, 9), cmap='coolwarm', title=factor_name)
    
    
    plt.figure(figsize=(16, 9))
    group_hist = [group_std[i].iloc[np.where(group_std[i].notna())[0][-1]] for i in range(num_group)]
    plt.bar(range(num_group), group_hist)
    plt.title(factor_name)
    
    group_r = {}
    for n in range(num_group):
        group_r[n] = (group_pos[n] * y).mean(1)
    DataFrame(group_r).plot(kind='kde', figsize=(16, 9), cmap='coolwarm', title=factor_name)
    
    
def generate_sql_y_x(factor_names, start_date, end_date, label_type='o', 
                     is_trade=True, is_industry=False, 
                     white_dic={'price': gc.LIMIT_PRICE, 'amount': gc.LIMIT_AMOUNT}, 
                     style_dic={'rank_mc': gc.LIMIT_RANK_MC}, 
                     n_ind=gc.LIMIT_N_IND,
                     n=gc.LIMIT_N):
    sql = """
    select t3.l3_name, t1.trade_date, t1.stock_code, 
    t1.r_{label_type} r, t1.r_a, t1.r_o, t1.r_c, 
    (ts.rank_mc) leading_stock, ts.mc style_mc, ts.pb style_pb
    """.format(label_type=label_type)
    
    for factor_name in factor_names:
        sql += ' , t{factor_name}.factor_value {factor_name} '.format(factor_name=factor_name)
    sql += ' from label.tdailylabel t1 '
    for factor_name in factor_names:
        sql += """ left join factor.tfactor{factor_name} t{factor_name} 
                   on t1.trade_date = t{factor_name}.trade_date 
                   and t1.stock_code = t{factor_name}.stock_code """.format(factor_name=factor_name)

    sql += """ 
    left join indsw.tindsw t3
    on t1.stock_code = t3.stock_code
    """
    sql += """
    left join style.tdailystyle ts
    on t1.trade_date = ts.trade_date 
    and t1.stock_code = ts.stock_code
    """
    
    sql += """ where t1.trade_date >= \'{start_date}\'
               and t1.trade_date <= \'{end_date}\'""".format(start_date=start_date, end_date=end_date)
    if is_trade:
        sql += " and t1.is_trade = 1 "
    
    if white_dic:
        for k in white_dic.keys():
            sql += " and t1.{k} >= {v} ".format(k=k, v=white_dic[k])
    if style_dic:
        for k in style_dic.keys():
            sql += " and ts.{k} >= {v} ".format(k=k, v=style_dic[k])
    if is_industry:
        sql += (" and t3.l3_name in %s "%gc.WHITE_INDUSTRY_LIST).replace('[', '(').replace(']', ')')
    
    if n_ind:
        sql = """
        select t.* from 
        (
        select t.*, rank() over (
            partition by trade_date, l3_name
            order by leading_stock desc
        ) my_rank_ind
        from 
        ({sql}) t
        ) t
        where t.my_rank_ind <= {n_ind}
        """.format(sql=sql, n_ind=n_ind)
    if n:
        sql = """
        select t.* from 
        (
        select t.*, rank() over (
            partition by trade_date 
            order by leading_stock desc
        ) my_rank
        from 
        ({sql}) t
        ) t
        where t.my_rank <= {n}
        """.format(sql=sql, n=n)
    
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


def generate_beta_alpha(data, factors=['beta', 'mc', 'bp'], ind='l1'):
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
        
        def generate_beta(data):
            if ind == None:
                X = data.loc[:, factors].fillna(0)
            else:
                X = pd.concat([pd.get_dummies(data.loc[:, ind]), data.loc[:, factors]], axis=1).fillna(0)
            
            X = standardize(X.T).T
            y = standardize(data.loc[:, 'y'])
            
            W = DataFrame(np.diag(data.loc[:, 'mc']), index=y.index, columns=y.index)
            beta = np.linalg.inv(X.T.dot(W).dot(X)+0.001*np.identity(len(X.T))).dot(X.T).dot(W).dot(y)
            y_predict = X.dot(beta)
            
            return DataFrame(y_predict)
        
        beta = data.groupby('trade_date').apply(generate_beta).iloc[:, 0]
        alpha = y - beta
        return beta.unstack(), alpha.unstack()
    else:
        return None
    

def neutralize(data, factors=['mc', 'bp'], ind='l3', ret_type='alpha'):
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
        df = pd.concat([y, df_n], axis=1).dropna()

        def g(df):
            if ind == None:
                X = df.loc[:, factors].fillna(0)
            else:
                X = pd.concat([pd.get_dummies(df.loc[:, ind]), df.loc[:, factors]], axis=1).fillna(0)
            
            y = df.loc[:, 'y']
            # W = DataFrame(np.diag(df.loc[:, 'mc']), index=y.index, columns=y.index)
            # y_predict = X.dot(np.linalg.inv(X.T.dot(W).dot(X)+0.001*np.identity(len(X.T))).dot(X.T).dot(W).dot(y))
            y_predict = X.dot(np.linalg.inv(X.T.dot(X)+0.001*np.identity(len(X.T))).dot(X.T).dot(y))
            
            return DataFrame(y - y_predict)
        x_n = df.groupby('trade_date').apply(g).iloc[:, 0]

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