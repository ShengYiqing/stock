from pandas import DataFrame
import tools

factors = [
    # 'value', 
    # 'quality', 
    'beta',
    'reversal', 
    'momentum',  
    'seasonality',
    'skew', 
    'cpshl', 
    'crhl', 
    'crhls', 
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
