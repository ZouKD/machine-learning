import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle

import utilities

# 从输入文件加载数据
X = utilities.load_data('data_multivar.txt')

# 设置带宽参数
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# 用Meanshift计算聚类
meanshift_estimator = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_estimator.fit(X)

# 提取标签
labels = meanshift_estimator.labels_

# 提取中心点
centroids = meanshift_estimator.cluster_centers_
num_clusters = len(np.unique(labels))
print('Number of clusters in input data = ', num_clusters)

# 画出数据点和聚类中心
plt.figure()

# 为每种集群设置不同的标记
markers = '.*xv'
for i, marker in zip(range(num_clusters), markers):
    # 画出属于某个急群中心的数据点
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='k')

    # 画出集群中心
    centroid = centroids[i]
    plt.plot(centroid[0], centroid[1], marker='o', markerfacecolor='k',
             markeredgecolor='k', markersize=15)

plt.title('Clusters and their centroids')
plt.show()

