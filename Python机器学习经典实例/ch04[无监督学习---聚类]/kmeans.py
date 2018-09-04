"""
用k-means算法聚类数据
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

import utilities

# 加载数据
data = utilities.load_data('data_multivar.txt')
num_clusters = 4

plt.figure()
plt.scatter(data[:,0], data[:,1], marker='o', facecolor='none', edgecolors='k', s=30)
x_min, x_max = min(data[:, 0]-1), max(data[:,0]+1)
y_min, y_max = min(data[:, 1]-1), max(data[:,1]+1)
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.xlim(y_min, y_max)
plt.xticks()
plt.yticks()

# 训练模型
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
kmeans.fit(data)

# 设置网格数据的步长
step_size = 0.01

# 画出边界
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# 预测网格中所有数据点的标记
predicted_labels = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()])

# 画出结果
predicted_labels = predicted_labels.reshape(x_values.shape)
plt.figure()
plt.clf()
plt.imshow(predicted_labels, interpolation='nearest', extent=(x_values.min(), x_values.max(),
                                                              y_values.min(), y_values.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')
plt.scatter(data[:,0], data[:,1], marker='o', facecolor='none', edgecolors='k', s=30)

# 把中心点画在图形上
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], marker='o', s=200, linewidths=3,
            color='k', zorder=10, facecolor='black')
plt.title('Centroids and boundaries obtained using KMeans')
plt.xlim(x_min, x_max)
plt.xlim(y_min, y_max)
plt.xticks()
plt.yticks()

plt.show()