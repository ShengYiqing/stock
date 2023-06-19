from pandas import DataFrame
import tools

f_list = ['cphl', 'cpv', 
           'crv', 'csrv', 'crsv', 
           'crhl', 'csrhl', 'crshl', 
           'cvhl', 'csvhl', 'cvshl', 
           'beta', 'str', 'hls', 'trd', 'sigmad']
f_list = ['cphl', 'cpv', 'crp', 
          'crsm',
          'crv', 'crsv', 
          'crhl', 'crshl', 
          'beta', 'str', 'hls', 'trd', 'sigmad'
          ]
f_list = ['crsm', 'wmsm', 'wmsmd']
corr_df = DataFrame(index=f_list, columns=f_list)

for i in range(len(f_list)):
    for j in range(len(f_list)):
        if j <= i:
            pass
        else:
            
            trade_date = '20230606'
            x1 = f_list[i]
            x2 = f_list[j]
            corr_df.loc[x1, x2] = tools.colinearity_analysis(x1, x2, trade_date)
            