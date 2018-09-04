from itertools import cycle

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

from utilities import load_data

# 加载输入数据
input_file = 'data_perf.txt'
X = load_data(input_file)

# 寻找最优的epsilon参数值
eps_grid = np.linspace(0.3, 1.2, num=10)
silhouette_scores = []
eps_best = eps_grid[0]
silhouette_score_max = -1
model_best = None
labels_best = None

for eps in eps_grid:
    # 训练DBSCAN聚类模型
    model = DBSCAN(eps=eps, min_samples=5).fit(X)
    # 提取标记
    labels = model.labels_
    # 提取性能指标
    silhouette_score = round(metrics.silhouette_score(X, labels), 4)
    silhouette_scores.append(silhouette_score)
    print('Epslion: ', eps, '--> silhouette score: ', silhouette_score)

    # 保存指标的最佳得分和对应的epsilon值
    if silhouette_score > silhouette_score_max:
        silhouette_score_max = silhouette_score
        eps_best = eps
        model_best = model
        labels_best = labels

# 画出条形图
plt.figure()
plt.bar(eps_grid, silhouette_scores, width=0.05, color='k', align='center')
plt.title('Silhouette score vs epsilon')

# 打印最优参数
print("Best epsilon = ", eps_best)

# 把最优的模型和标记保存起来
# 最优参数对应的模型于标记
model = model_best
labels = labels_best

# 检查标记中没有分配集群的数据点
offset = 0
if -1 in labels:
    offset = 1

# 数据中的集群数量
num_clusters = len(set(labels)) - offset
print("Estimated number of clusters = ", num_clusters)

# 从训练模型中提取合兴样本的数据点索引
mask_core = np.zeros(labels.shape, dtype=np.bool)
mask_core[model.core_sample_indices_] = True

# 画出集群结果
plt.figure()
labels_uniq = set(labels)
markers = cycle('vo^s<>')

for cur_label, marker in zip(labels_uniq, markers):
    # 用黑点表示未分配的数据点
    if cur_label == -1:
        marker = '.'

    # 为当前标记添加符号
    cur_mask = (labels == cur_label)
    cur_data = X[cur_mask & mask_core]
    plt.scatter(cur_data[:,0], cur_data[:,1], marker=marker, edgecolors='black',
                s=96, facecolor='none')
    cur_data = X[cur_mask & ~mask_core]
    plt.scatter(cur_data[:,0], cur_data[:,1], marker=marker, edgecolors='black',
                s=32)

plt.title('Data separated into clusters')
plt.show()

