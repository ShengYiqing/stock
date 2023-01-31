import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import Config
sys.path.append(Config.GLOBALCONFIG_PATH)
import tools
import Global_Config as gc
import multiprocessing as mp

def f(factor_file, start_date, end_date):
    print(factor_file)
    exec('from %s import generate_factor as %s_generate_factor'%(factor_file.split('.')[0], factor_file.split('.')[0]))
    exec('%s_generate_factor(start_date, end_date)'%factor_file.split('.')[0])

if __name__ == '__main__':
    end_date = datetime.datetime.today().strftime('%Y%m%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(15)).strftime('%Y%m%d')
    # start_date = '20220930'
    files = os.listdir('D:/stock/Codes/Factor/')
    factor_files = list(filter(lambda x:x[-4]>='0' and x[-4]<='9', files))
    factor_files_dic = {}
    for i in range(10):
        factor_files_dic[i] = list(filter(lambda x:x[-4]==str(i), files))
    
    for i in range(10):
        factor_files_i = factor_files_dic[i]
        pool = mp.Pool(4)
        for factor_file in factor_files_i:
            pool.apply_async(func=f, args=(factor_file, start_date, end_date))
            # exec('from %s import generate_factor as %s_generate_factor'%(factor_file.split('.')[0], factor_file.split('.')[0]))
            # exec('%s_generate_factor(start_date, end_date)'%factor_file.split('.')[0])
            # os.system('python D:/stock/Factor/Codes/%s'%factor_file)
        pool.close()
        pool.join()