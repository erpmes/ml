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


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np
import graphviz



# 加载鸢尾花数据集
iris = load_iris()

# 查看数据集的特征矩阵
print("特征矩阵:")
print(iris.data)

# 查看数据集的目标向量
print("目标向量:")
print(iris.target)

# 查看特征名称
print("特征名称:")
print(iris.feature_names)

# 查看目标类别名称
print("目标类别名称:")
print(iris.target_names)


X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier(max_depth=100)

# 在训练集上训练决策树模型
clf.fit(X_train, y_train)

# Access the decision tree object of regr_1
tree_1 = clf.tree_

# Access the feature and threshold of each node in the tree
features = tree_1.feature
thresholds = tree_1.threshold

def get_y_label(tree, node_index):
    if tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1:
        # Leaf node
        value = tree.value[node_index]
        class_index = np.argmax(value)
        y_label = iris.target_names[class_index]
        return y_label
    else:
        # Internal node
        return None

# Print the decision rules of each node
for i in range(tree_1.node_count):
    if tree_1.children_left[i] == -1 and tree_1.children_right[i] == -1:
        # Leaf node
        print(f"Node {i}: Predicted value = {tree_1.value[i]}, y_label = {get_y_label(tree_1, i)}")
        
    else:
        # Internal node
        feature = features[i]
        threshold = thresholds[i]
        print(f"Node {i}: If feature {feature} <= {threshold}, go to node {tree_1.children_left[i]}, else go to node {tree_1.children_right[i]}")

# 可视化决策树
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,  
                                filled=True, rounded=True,  
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")  # 保存为PDF文件
graph.view()  # 在默认PDF阅读器中打开


# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
print("模型的准确率:", accuracy)




