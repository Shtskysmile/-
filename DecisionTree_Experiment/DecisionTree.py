import math
from queue import Queue
import matplotlib.pyplot as plt

class DecisionTreeClassifier(object):
    class Node(object):
        def __init__(self):
            self.left = None    # 左子树
            self.right = None   # 右子树
            self.feature = None # 划分特征
            self.threshold = None   # 划分阈值
            self.depth = None   # 深度
            self.y = None   # 叶子节点的类别

        def split(self, X, y):
            left_x = []
            left_y = []
            right_x = []
            right_y = []
            for i in range(len(X)):
                if X[i][self.feature] < self.threshold:
                    left_x.append(X[i])
                    left_y.append(y[i])
                else:
                    right_x.append(X[i])
                    right_y.append(y[i])

            return left_x, left_y, right_x, right_y

    def __init__(self, max_depth=5, min_sample=1):
        self.root = self.Node()
        self.max_depth = max_depth
        self.min_sample = min_sample

    def information_entropy(self, y):
        """
        计算信息熵函数
        :param y: 标签集合
        :return: 信息熵
        """
        entropy = 0
        for label in set(y):
            p = sum(1 for i in y if i == label) / len(y)
            entropy -= p * math.log2(p)
        return entropy

    def information_gain(self, X, y, feature, threshold):
        left_y = []
        right_y = []
        for i in range(len(X)):
            if X[i][feature] < threshold:
                left_y.append(y[i])
            else:
                right_y.append(y[i])

        return self.information_entropy(y) \
            - len(left_y) / len(y) * self.information_entropy(left_y) \
            - len(right_y) / len(y) * self.information_entropy(right_y)


    def _fit(self, node, X, y, depth):
        if depth >= self.max_depth or len(set(y)) <= self.min_sample:
            node.y = max(set(y), key=y.count)
            return

        max_gain = 0
        for feature in range(len(X[0])):
            temp_X_feature = [X[i][feature] for i in range(len(X))]
            temp_X_feature.sort()
            for idx in range(len(temp_X_feature) - 1):
                threshold = (temp_X_feature[idx] + temp_X_feature[idx + 1]) / 2
                gain = self.information_gain(X, y, feature, threshold)
                if gain > max_gain:
                    max_gain = gain
                    node.feature = feature
                    node.threshold = threshold

        left_x, left_y, right_x, right_y = node.split(X, y)
        node.left = self.Node()
        node.right = self.Node()
        self._fit(node.left, left_x, left_y, depth + 1)
        self._fit(node.right, right_x, right_y, depth + 1)

    def fit(self, X, y):
        """
        训练模型，对外的接口
        :param X: 特征
        :param y: 标签
        :return: None
        """
        self._fit(self.root, X, y, 0)

    def _predict(self, node, x):
        if node.feature is None:
            return node.y

        if x[node.feature] < node.threshold:
            return self._predict(node.left, x)

        else:
            return self._predict(node.right, x)

    def predict(self, X):
        """
        预测接口
        :param X: 样本特征
        :return: 结果
        """
        return [self._predict(self.root, x) for x in X]

    def print_tree(self):
        """
        打印决策树，使用bfs实现
        :return: None
        """
        q = Queue()
        q.put(self.root)
        while not q.empty():
            node = q.get()
            if node.feature is None:
                print('#' * 20)
                print('leaf node: {}'.format(node.y))
                print('#' * 20)
                print()
                continue

            print('#' * 20)
            print('feature: {}, threshold: {}, depth: {}'.format(node.feature, node.threshold, node.depth))
            print('#' * 20)
            print()
            if node.left is not None:
                q.put(node.left)

            if node.right is not None:
                q.put(node.right)

    def plot_tree(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_node(ax, self.root, 0.5, 1, 0.5, 0)
        plt.show()

    def _plot_node(self, ax, node, x, y, dx, depth):
        if node.feature is None:
            # 绘制非叶子节点
            ax.text(x, y, f'Class: {node.y}', ha='center', va='center',
                    bbox=dict(facecolor='lightgreen', edgecolor='black'))
        else:
            # 绘制叶子节点
            ax.text(x, y, f'Feature: {node.feature}\nThreshold: {node.threshold}', 
                    ha='center', 
                    va='center',
                    bbox=dict(facecolor='lightblue', edgecolor='black'))

            if node.left:
                # 绘制边 k-画黑色实线
                ax.plot([x, x - dx], [y - 0.1, y - 0.2], 'k-')
                self._plot_node(ax, node.left, x - dx, y - 0.3, dx / 1.5, depth + 1)
            if node.right:
                # 绘制边
                ax.plot([x, x + dx], [y - 0.1, y - 0.2], 'k-')
                self._plot_node(ax, node.right, x + dx, y - 0.3, dx / 1.5, depth + 1)


