'''
鸢尾花数据集（Iris dataset）有4个特征（features），它们分别是：

花萼长度（sepal length）：以厘米（cm）为单位。
花萼宽度（sepal width）：以厘米（cm）为单位。
花瓣长度（petal length）：以厘米（cm）为单位。
花瓣宽度（petal width）：以厘米（cm）为单位。

鸢尾花数据集（Iris dataset）有3个标签（labels），它们分别是：

Setosa（山鸢尾花）
Versicolor（变色鸢尾花）
Virginica（维吉尼亚鸢尾花）
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data

# 创建一个列表来保存不同聚类数量对应的轮廓系数
silhouette_scores = []

# 尝试不同的聚类数量
for n_clusters in range(2, 11):
    # 创建k-means模型
    kmeans = KMeans(n_clusters=n_clusters)

    # 对数据进行聚类
    kmeans.fit(X)

    # 获取聚类结果
    labels = kmeans.labels_

    
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    
    # 绘制聚类结果
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('K-means Clustering{n_clusters}')
    plt.show()

    # 计算轮廓系数
    score = silhouette_score(X, labels)

    # 将轮廓系数添加到列表中
    silhouette_scores.append(score)

# 绘制轮廓系数与聚类数量的关系图
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient vs Number of Clusters')
plt.show()
