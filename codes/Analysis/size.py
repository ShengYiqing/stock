import pandas as pd
from pandas import Series, DataFrame
from sqlalchemy import create_engine
import tools

engine = create_engine("mysql+pymysql://root:12345678@127.0.0.1:3306/")
sql = """
select stock_code, (total_mv / total_share * free_share) mc
from tsdata.ttsdailybasic
where trade_date = '20231117'
"""

df = pd.read_sql(sql, engine)
s = df.set_index('stock_code').mc.sort_values(ascending=False)

s = s / s.sum()
s = s.cumsum()
s.plot()
s.iloc[:3800]
