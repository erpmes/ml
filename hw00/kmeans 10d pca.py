#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 11:45:37 2023

@author: gaffliu

鸢尾花数据集（Iris dataset）有4个特征（features），它们分别是：

花萼长度（sepal length）：以厘米（cm）为单位。
花萼宽度（sepal width）：以厘米（cm）为单位。
花瓣长度（petal length）：以厘米（cm）为单位。
花瓣宽度（petal width）：以厘米（cm）为单位。

鸢尾花数据集（Iris dataset）有3个标签（labels），它们分别是：

Setosa（山鸢尾花）
Versicolor（变色鸢尾花）
Virginica（维吉尼亚鸢尾花）

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 生成高维数据集
X, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)

# 使用PCA将数据降维到三维
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 创建k-means模型
kmeans = KMeans(n_clusters=3)

# 对降维后的数据进行聚类
kmeans.fit(X_pca)

# 获取聚类结果
labels = kmeans.labels_

# 绘制聚类结果
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('K-means Clustering (PCA)')
plt.show()
