print('HelloWorld')
print(HelloWorld) #HelloWorld被识别为未定义的对象而不是字符串


for i in range(10):
print(10) #没有缩进


import numpy as np
import pandas as pd
df = pd.DataFrame(np.arange(10))
s = df.loc[:, '0'] #列名是数字0不是字符串'0'


dic = []
keys = dic.keys() #keys()方法是dict类型的方法，但是变量dic被定义为了list类型，不具备keys()方法