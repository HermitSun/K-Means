import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# 聚类的簇数
n_clusters = 8
# 偏差，小于该值则视为聚类已完成
bias = 0.0001


# 随机初始化center
def init_center(data):
    n_sample, n_feature = data.shape

    f_mean = np.mean(data)
    f_std = np.std(data)

    cluster_centers = f_mean + np.random.randn(n_clusters, n_feature) * f_std
    return cluster_centers


def k_means(data):
    n_sample, n_feature = data.shape

    # 随机初始化center
    cluster_centers = init_center(data)
    dist = np.zeros((n_sample, n_clusters))
    current_center = np.zeros(cluster_centers.shape)
    center_move = np.linalg.norm(cluster_centers - current_center)
    # 类别
    labels = []

    while center_move > bias:
        # 计算距离
        for i in range(n_clusters):
            dist[:, i] = np.linalg.norm(data - cluster_centers[i], axis=1)
        # 标注类别
        labels = np.argmin(dist, axis=1)

        # 因为py的语言特性（默认浅拷贝），需要进行深拷贝
        current_center = deepcopy(cluster_centers)
        # 展示坐标变化
        print(current_center)

        # 更新center
        for i in range(n_clusters):
            cluster_centers[i] = np.mean(data[labels == i], axis=0)
        # 计算移动距离
        center_move = np.linalg.norm(cluster_centers - current_center)

    return cluster_centers, labels


# 固定随机数
np.random.seed(7)

# 进行聚类
data = np.random.randn(100, 2)
centers, categories = k_means(data)

# 绘图
plt.clf()
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, c=categories)
plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='k')
plt.show()
