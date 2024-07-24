#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:55:37 2024

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化AdaBoost分类器，使用决策树桩作为弱分类器
ada_boost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42, algorithm='SAMME')

# 训练AdaBoost分类器
ada_boost.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = ada_boost.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 绘制分类结果
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
markers = ['o', 's', 'x']

for target_class in np.unique(y):
    plt.scatter(X_test[y_test == target_class, 0], X_test[y_test == target_class, 1],
                marker=markers[target_class], color=colors[target_class], label=target_class)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("AdaBoost Classification Results")
plt.legend()
plt.show()
