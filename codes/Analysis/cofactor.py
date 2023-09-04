from pandas import DataFrame
import tools

factors = [
    'beta', 
    'hfbeta' 
    ]
c = {}

for i in range(len(factors)):
    for j in range(len(factors)):
        if j <= i:
            pass
        else:
            
            start_date = '20220101'
            end_date = '20240101'
            x1 = factors[i]
            x2 = factors[j]
            c[x1+'*'+x2] = tools.colinearity_analysis(x1, x2, start_date, end_date)
DataFrame(c).plot()
