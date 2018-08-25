import pandas as pd
import numpy as np
import os
os.chdir(r'E:\machine-learning\航空公司客户价值分析\air_data')

data = pd.read_csv('air_data.csv', encoding='utf-8')
data.shape
data_descirbe = data.describe().T
data_descirbe.head(10)
# 计算空值
