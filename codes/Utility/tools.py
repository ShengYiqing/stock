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
    plt.scatter(df.iloc[:, 0].rank(), df.iloc[:, 1].rank())
    print(df.corr())

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
    IC.abs().cumsum().plot()
    plt.title(factor_name+'-ic abs')
    # plt.legend(['%s'%i for i in range(len(ys))])
    # plt.savefig('%s/Results/%s/IC_abs.png'%(gc.SINGLEFACTOR_PATH, self.factor_name))
        
        
    x_quantile = DataFrame(x.rank(axis=1)).div(x.notna().sum(1), axis=0)
    
    group_pos = {}
    for n in range(num_group):
        group_pos[n] = DataFrame((n/num_group <= x_quantile) & (x_quantile <= (n+1)/num_group))
        group_pos[n][~group_pos[n]] = np.nan
        group_pos[n] = 1 * group_pos[n]
            
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
    
def generate_context_factor(factor, context, start_date, end_date, pieces=3):
    sql = """
    select t1.trade_date, t1.stock_code, t1.factor_value {factor}, t2.factor_value {context} from tfactor{factor} t1 
    left join tfactor{context} t2 
    on t1.stock_code = t2.stock_code 
    and t1.trade_date = t2.trade_date 
    where t1.trade_date >= {start_date} 
    and t1.trade_date <= {end_date} """.format(factor=factor, context=context, start_date=start_date, end_date=end_date)
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    
    df = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code'])
    factor = df.loc[:, factor].unstack()
    context = df.loc[:, context].unstack()
    
    context_quantile = context.rank(axis=1).div(context.notna().sum(1), axis=0)
    ret = {}
    for i in range(pieces):
        factor_tmp = factor.copy()
        factor_tmp[~((i/pieces <= context_quantile)&(context_quantile <= (i+1)/pieces))] = 0
        ret[i] = factor_tmp
    
    return ret


def generate_high_order_factor(formula, start_date, end_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factor?charset=utf8")
    
    formula = formula.replace('\n', ' ').replace(' ', '')
    subjects = formula.replace('(', ' ').replace(')', ' ').replace('+', ' ').replace('-', ' ').replace('*', ' ').replace('/', ' ').split(' ')
    subjects = list(filter(lambda x:x!= '', subjects))
    subjects = list(filter(lambda x:not x.isdigit(), subjects))
    
    n = len(subjects)
    tables = [subject.split('.')[0] for subject in subjects]
    values = [subject.split('.')[1] for subject in subjects]
    
    stock_codes = []
    for i in range(n):
        sql = """
        select trade_date, stock_code, {value} as v from {table} 
        where trade_date >= {start_date} 
        and trade_date <= {end_date}
        """.format(value=values[i], table=tables[i], start_date=start_date, end_date=end_date)
        exec("""%s = pd.read_sql(sql, engine).set_index(['trade_date', 'stock_code']).unstack().loc[:, 'v']"""%tables[i])
        exec('stock_codes.extend(list(%s.columns))'%tables[i])
        stock_codes = list(set(stock_codes))
    stock_codes = sorted(stock_codes)
    for i in range(n):
        exec("""%s = DataFrame(%s, columns=stock_codes).fillna(0)"""%(tables[i], tables[i]))
    
    factor = eval(formula.replace('.factor_value', '').replace('.preprocessed_factor_value', ''))
    return factor

def generate_finance_formula(formula, start_date, end_date, shift=2000, method=None):
    if shift < 2000:
        shift = 2000
    start_date_data = trade_date_shift(start_date, shift)
    
    formula = formula.replace(' ', '')
    subjects = formula.replace('(', ' ').replace(')', ' ').replace('+', ' ').replace('-', ' ').replace('*', ' ').replace('/', ' ').split(' ')
    subjects = list(filter(lambda x:x!= '', subjects))
    
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    
    stock_codes = []
    subject_df_dic = {}
    for table_subject in subjects:
        table = table_subject.split('.')[0]
        subject = table_subject.split('.')[1]
        sql = """
        select stock_code, end_date, f_ann_date, update_flag, {subject} from  
        """.format(subject=subject)
        if table == 't1':
            sql = sql + """ ttsbalancesheet """
        if table == 't2':
            sql = sql + """ ttsincome """
        if table == 't3':
            sql = sql + """ ttscashflow """
        sql = sql + """ where end_date >= \'{start_date_data}\' """.format(start_date_data=start_date_data)
        sql = sql + """ and substr(end_date, 5) in ('0331', '0630', '0930', '1231') """
        sql = sql + """ and comp_type = '1' """
        subject_df_dic[subject] = pd.read_sql(sql, engine)
        stock_codes.extend(list(subject_df_dic[subject].stock_code))
        stock_codes = list(set(stock_codes))
    stock_codes = sorted(stock_codes)
    trade_cal = get_trade_cal(start_date, end_date)
    factor = DataFrame(index=trade_cal, columns=stock_codes)
    for trade_date in trade_cal:
        print(trade_date)
        trade_date_start = trade_date_shift(trade_date, shift)
        
        for table_subject in subjects:
            table = table_subject.split('.')[0]
            subject = table_subject.split('.')[1]
            df = subject_df_dic[subject]
            df = df.loc[(df.f_ann_date<=trade_date)&(df.end_date>=trade_date_start), :]
            df = df.sort_values(['stock_code', 'end_date', 'f_ann_date', 'update_flag']).groupby(['stock_code', 'end_date']).last()
            df_tmp = df.loc[:, subject].unstack(['stock_code'])
            if table == 't1':
                df_tmp = df_tmp.rolling(2, min_periods=1).mean()
            else:
                cols = df_tmp.columns
                df_tmp['YYYY'] = [ind[:4] for ind in df_tmp.index]
                df_tmp = df_tmp.groupby('YYYY').apply(lambda x:(x.sub(x.shift().fillna(0, limit=1))))
                df_tmp = df_tmp.loc[:, cols]
            exec("""%s = DataFrame(df_tmp, columns=stock_codes).fillna(0)"""%subject)
        
        factor_tmp = eval(formula.replace('t1.', '').replace('t2.', '').replace('t3.', ''))
        factor_tmp.replace(0, np.nan, inplace=True)
        
        if method == None or method == '':
            factor_tmp = factor_tmp.rolling(20, min_periods=4).mean()
        if method == 'd':
            factor_tmp = factor_tmp.rolling(4, min_periods=1).mean().diff(20)
        if method == 's':
            factor_tmp = factor_tmp.rolling(20, min_periods=4).std()
        factor_tmp.fillna(method='ffill', limit=1, inplace=True)
        
        factor.loc[trade_date, :] = factor_tmp.iloc[-1,:]
    factor = factor.astype(float)
    return factor
    
def generate_sql_y_x(factor_names, start_date, end_date, is_trade=True, is_white=True, is_industry=True, factor_value_type_dic=None, y_neutral=False):
    if factor_value_type_dic == None:
        factor_value_type_dic = {factor_name: 'preprocessed_factor_value' for factor_name in factor_names}
    if y_neutral:
        sql = ' select t1.trade_date, t1.stock_code, t1.neutral_r_daily r_daily, t1.neutral_r_weekly r_weekly, t1.neutral_r_monthly r_monthly '
    else:
        sql = ' select t1.trade_date, t1.stock_code, t1.r_daily, t1.r_weekly, t1.r_monthly '
    for factor_name in factor_names:
        sql += ' , t{factor_name}.{factor_value_type} {factor_name} '.format(factor_name=factor_name, factor_value_type=factor_value_type_dic[factor_name])
    sql += ' from label.tdailylabel t1 '
    for factor_name in factor_names:
        sql += """ left join factor.tfactor{factor_name} t{factor_name} 
                   on t1.trade_date = t{factor_name}.trade_date 
                   and t1.stock_code = t{factor_name}.stock_code """.format(factor_name=factor_name)
    if is_white:
        sql += """ left join whitelist.tdailywhitelist t2
                   on t1.trade_date = t2.trade_date
                   and t1.stock_code = t2.stock_code """
    if is_industry:
        sql += """ left join indsw.tindsw t3
                   on t1.stock_code = t3.stock_code """
    sql += """ where t1.trade_date >= \'{start_date}\'
               and t1.trade_date <= \'{end_date}\'""".format(start_date=start_date, end_date=end_date)
    if is_trade:
        sql += " and t1.is_trade = 1 "
    if is_white:
        sql += " and t2.white = 1 "
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
  
def neutral_apply(y, x_list):
    try:
        date = y.name
        X = DataFrame(index=y.index)
        for x in x_list:
            x = DataFrame(x, columns=y.index)
            X = pd.concat([X, x.loc[date, y.index]], axis=1)
        X.fillna(0, inplace=True)
        X = sm.add_constant(X)
        res = y - X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y.fillna(0)))
    except:
        print(date)
    return res

def get_trade_cal(start_date, end_date):
    engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/tsdata?charset=utf8")
    sql_trade_cal = """
    select distinct cal_date from ttstradecal where is_open = 1
    """
    
    trade_cal = list(pd.read_sql(sql_trade_cal, engine).loc[:, 'cal_date'])
    trade_cal = list(filter(lambda x:(x>=start_date) & (x<=end_date), trade_cal))
    return trade_cal

def get_industrys(level='L1', stocks=None):
    #获取行业分类
    file_list = os.listdir(gc.DATABASE_PATH+'/StockIndustryData')
    file_list.sort()
    file = file_list[-1]
    df = pd.read_csv(gc.DATABASE_PATH+'/StockIndustryData/%s'%file, dtype=str)
    df.dropna(inplace=True)
    
    ind_name_code_dict = {k:k for k in set(df.loc[:,'行业名称'])}
    industrys = {}
    for ind_name in ind_name_code_dict.keys():
        industrys[ind_name_code_dict[ind_name]] = list(df.loc[df.loc[:,'行业名称']==ind_name, '股票代码'])
        industrys[ind_name_code_dict[ind_name]] = [stock + '.SZ' if (stock[0]=='0' or stock[0]=='3') else stock + '.SH' for stock in industrys[ind_name_code_dict[ind_name]]]
    
    if stocks:
        def cond(stock):
            return stock in stocks
        industrys = {i:list(filter(cond, industrys[i])) for i in industrys.keys()}
    
        stocks.clear()
        for v in industrys.values():
            stocks.extend(v)
        stocks.sort()
    return industrys

def reg_ts(df, n):
    x = np.arange(n)
    x = x - x.mean()
    b = df.rolling(n).apply(lambda y:(y*x).sum() / (x*x).sum(), raw=True)
    a = df.rolling(n).mean()
    y_hat = a + b * x[-1]
    e = df - y_hat
    
    return b, e

def neutralize(data, factors=['mc']):
    if isinstance(data, DataFrame):
        data.index.name = 'trade_date'
        data.columns.name = 'stock_code'
        engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/?charset=utf8")
        f0 = factors[0]
        sql = """
        select t{f0}.trade_date trade_date, t{f0}.stock_code stock_code, t{f0}.preprocessed_factor_value {f0}
        """.format(f0=f0)
        if len(factors) > 1:
            for f in factors[1:]:
                sql += """
                , t{f}.preprocessed_factor_value {f}
                """.format(f=f)
        sql += """
        , tind.ind_code ind 
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
        
        sql += """
        left join indsw.tindsw tind
        on t{f0}.stock_code = tind.stock_code
        where t{f0}.trade_date in {trade_dates}
        and t{f0}.stock_code in {stock_codes}
        """.format(f0=f0, trade_dates=tuple(data.index), stock_codes=tuple(data.columns))
        
        df_n = pd.read_sql(sql, engine)
        df_n = df_n.set_index(['trade_date', 'stock_code'])
        x = data.stack()
        x.name = 'x'
        data = pd.concat([x, df_n], axis=1).dropna()

        def g(data):
            # pdb.set_trace()
            X = pd.concat([pd.get_dummies(data.ind), data.loc[:, factors]], axis=1).fillna(0)
            # X = data.loc[:, ['mc', 'bp']]
            # print(X)
            y = data.loc[:, 'x']
            # model = LinearRegression(n_jobs=-1)
            # model.fit(X, y)
            # y_predict = Series(model.predict(X), index=y.index)
            y_predict = X.dot(np.linalg.inv(X.T.dot(X)+0.01*np.identity(len(X.T))).dot(X.T).dot(y))
            res = standardize(winsorize(y - y_predict))
            return res
        x_n = data.groupby('trade_date', as_index=False).apply(g).reset_index(0, drop=True)
        # x_n.name = 'neutral_factor_value'
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
    
def standardize_industry(data, industrys=None):
    if industrys == None:
        industrys = get_industrys('L1', list(data.columns))
    data_dic = {k:standardize(DataFrame(data, columns=industrys[k])) for k in industrys.keys()}
    ret = pd.concat([df for df in data_dic.values()], axis=1)
    
    return ret

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