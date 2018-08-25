import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import neighbors, datasets
import pandas as pd

# 加载数据
data = pd.read_table("data_nn_classifier.txt", sep=',', header=None)
# print(data)
X, y =data.iloc[:,:-1], data.iloc[:,-1]
# print(X)
# print(y)

# 画出输入数据
plt.figure(1)
plt.title("Input datapoints")
markers = '^sov<>hp'
mapper = np.array([markers[i] for i in y])
for i in range(X.shape[0]):
    plt.scatter(X.iloc[i,0], X.iloc[i,1], marker=mapper[i], s=50, edgecolors='black', facecolor='none')

# 设置最近邻个数
num_neighbors = 10

# 设置网格步长
h = 0.01

# 构建KNN模型并训练
classifier = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')
classifier.fit(X, y)

# 建立网格来画出边界
x_min, x_max = X.iloc[:,0].min()-1, X.iloc[:,0].max()+1
y_min, y_max = X.iloc[:,1].min()-1, X.iloc[:,1].max()+1
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 计算网格所有点的输出
predicted_values = classifier.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

# 画出计算结果
predicted_values = predicted_values.reshape(x_grid.shape)
plt.figure(2)
plt.pcolormesh(x_grid, y_grid, predicted_values, cmap=cm.Pastel1)  # cm.Pastel1使背景颜色柔和点

# 在图上画上训练点
for i in range(X.shape[0]):
    plt.scatter(X.iloc[i,0], X.iloc[i,1], marker=mapper[i], s=50, edgecolors='black', facecolor='none')
plt.xlim(x_grid.min(), x_grid.max())
plt.ylim(y_grid.min(), y_grid.max())
plt.title('k nearest neighbors classifier boundaries')

# 画出测试点
test_datapoint = [[4.5, 3.6]]
plt.figure(3)
plt.title('Test datapoint')
for i in range(X.shape[0]):
    plt.scatter(X.iloc[i,0], X.iloc[i,1], marker=mapper[i], s=50, edgecolors='black', facecolor='none')
plt.scatter(test_datapoint[0][0], test_datapoint[0][1], marker='x', linewidths=3, s=200, facecolor='black')

# 提取KNN分类结果
dist, indices = classifier.kneighbors(test_datapoint)

# 画出KNN分类结果
plt.figure(4)
plt.title('k nearest neighbors')

for i in indices:
    plt.scatter(X.iloc[i,0], X.iloc[i,1], linewidths=3, s=100, facecolor='black')

plt.scatter(test_datapoint[0][0], test_datapoint[0][1], marker='x', linewidths=3, s=200, facecolor='black')

for i in range(X.shape[0]):
    plt.scatter(X.iloc[i,0], X.iloc[i,1], marker=mapper[i],
            s=50, edgecolors='black', facecolors='none')

print("Predicted output:", classifier.predict(test_datapoint)[0])

plt.show()