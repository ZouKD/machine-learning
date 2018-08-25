import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 输入数据
X = np.array([[1, 1], [1, 3], [2, 2], [2.5, 5], [3, 1],
        [4, 2], [2, 3.5], [3, 3], [3.5, 4]])

# 查找最近邻的数据
num_neighbors = 3

# 随机输入数据点
input_point = [[2.6, 1.7]]

# 画出数据点
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', s=25, color='r')

# 建立最近邻模型
knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)
distances, indices = knn.kneighbors(input_point)
print("distances:", distances)
print("indices: ", indices)

# 打印k个最近邻点
print("k nearest neighbors")
for rank, index in enumerate(indices[0][:num_neighbors]):
    print(str(rank+1) + "-->", X[index])

# 画出最近邻
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', color='b')
plt.scatter(X[indices][0][:][:,0], X[indices][0][:][:,1], marker='o', s=150,
            color='b', facecolor='none')
plt.scatter(input_point[0][0], input_point[0][1], marker='x', s=150, color='r')
plt.show()
