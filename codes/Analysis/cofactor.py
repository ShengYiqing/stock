from pandas import DataFrame
import tools

factors = ['mc', 
           'bp', 
           'reversal', 
           'tr', 
           'dailytech',
           'hftech', 
           ]
r2_dic = {}

for i in range(len(factors)):
    for j in range(len(factors)):
        if j <= i:
            pass
        else:
            
            start_date = '20230101'
            end_date = '20240101'
            x1 = factors[i]
            x2 = factors[j]
            if x1 == 'dailytech' or x2 == 'hftech':
                r2_dic[x1+'*'+x2] = tools.colinearity_analysis(x1, x2, start_date, end_date)
DataFrame(r2_dic).plot()
