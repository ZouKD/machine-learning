import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
X = iris.data
Y = iris.target
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['species'] = df['species'].map({'setosa':0, 'versicolor':1, 'virginica':2})
data = df.values

#划分训练集和测试集
X, y=np.split(data,(4,),axis=1) #这里用大写X， np.split 按照列（axis=1）进行分割
x_train,x_test,y_train,y_test=model_selection.train_test_split(X,y,random_state=1,test_size=0.3)


#搭建模型，训练GaussianNB分类器
clf=GaussianNB()

#开始训练
clf.fit(x_train,y_train)
y_hat = clf.predict(x_test)
# print(y_hat)
print(clf.score(x_test, y_test))

# 计算GaussianNB分类器的准确率
print("GaussianNB-输出训练集的准确率为：", clf.score(x_train, y_train))
y_hat = clf.predict(x_train)

print("GaussianNB-输出测试集的准确率为：", clf.score(x_test, y_test))
y_hat = clf.predict(x_test)




