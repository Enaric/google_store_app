import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('googleplaystore_clean_val.csv')
examDf = DataFrame(data)

# 数据清洗,比如第一列有可能是日期，这样的话我们就只需要从第二列开始的数据，
# 这个情况下，把下面中括号中的0改为1就好，要哪些列取哪些列
# new_examDf = examDf.ix[:, 0:]

# 检验数据
print(examDf.describe())  # 数据描述，会显示最值，平均数等信息，可以简单判断数据中是否有异常值
print(examDf[examDf.isnull() == True].count())  # 检验缺失值，若输出为0，说明该列没有缺失值

print("输出相关系数，判断是否值得做线性回归模型")
# 输出相关系数，判断是否值得做线性回归模型
print(examDf.corr())  # 0-0.3弱相关；0.3-0.6中相关；0.6-1强相关；

# 通过seaborn添加一条最佳拟合直线和95%的置信带，直观判断相关关系
sns.pairplot(data, x_vars=['Category Val', 'Reviews','Installs','Size','Type Val','Price'], y_vars='Rating', height=7, aspect=0.8, kind='reg')
# sns.pairplot(data, x_vars=['Reviews'], y_vars='Rating', height=7, aspect=0.8, kind='reg')

plt.show()