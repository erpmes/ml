#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:28:31 2023

鸢尾花数据集（Iris dataset）有4个特征（features），它们分别是：

花萼长度（sepal length）：以厘米（cm）为单位。
花萼宽度（sepal width）：以厘米（cm）为单位。
花瓣长度（petal length）：以厘米（cm）为单位。
花瓣宽度（petal width）：以厘米（cm）为单位。

鸢尾花数据集（Iris dataset）有3个标签（labels），它们分别是：

Setosa（山鸢尾花）
Versicolor（变色鸢尾花）
Virginica（维吉尼亚鸢尾花）

@author: gaffliu
"""

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

print(X)

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=10)
regr_2 = DecisionTreeRegressor(max_depth=3)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Access the decision tree object of regr_1
tree_1 = regr_1.tree_

# Access the feature and threshold of each node in the tree
features = tree_1.feature
thresholds = tree_1.threshold

# Print the decision rules of each node
for i in range(tree_1.node_count):
    if tree_1.children_left[i] == -1 and tree_1.children_right[i] == -1:
        # Leaf node
        print(f"Node {i}: Predicted value = {tree_1.value[i]}")
    else:
        # Internal node
        feature = features[i]
        threshold = thresholds[i]
        print(f"Node {i}: If feature {feature} <= {threshold}, go to node {tree_1.children_left[i]}, else go to node {tree_1.children_right[i]}")


# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()