from pandas import DataFrame
import pandas as pd
import tools
from sqlalchemy import create_engine

factors = [
    'beta', 
    'mc', 
    'bp', 
    'jump', 
    'reversal', 
    'momentum', 
    'seasonality', 
    'skew', 
    'crhl', 
    'cphl', 
    'quality'
    ]

c = {}

for i in range(len(factors)):
    for j in range(len(factors)):
        if j <= i:
            pass
        else:
            
            start_date = '20230101'
            end_date = '20240101'
            x1 = factors[i]
            x2 = factors[j]
            c[x1+'*'+x2] = tools.colinearity_analysis(x1, x2, start_date, end_date)
df = DataFrame(c)
df.to_csv('C:/Users/admin/Desktop/因子相关性.csv')
df.plot()


            
start_date = '20230101'
end_date = '20240101'
engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/factorevaluation?charset=utf8")

sql_ic = """
select trade_date, factor_name, 
ic ic
from tdailyfactorevaluation
where factor_name in {factors}
and trade_date >= {start_date}
and trade_date <= {end_date}
"""
sql_ic = sql_ic.format(factors=tuple(factors), start_date=start_date, end_date=end_date)
df_ic = pd.read_sql(sql_ic, engine).set_index(['trade_date', 'factor_name']).ic.unstack()
ic_corr = df_ic.corr()
