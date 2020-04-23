# -*- encoding:utf-8 -*-
# @Time    : 2020/4/22 09:50 上午
# @Author  : HengGeZhiZou <1018676477@qq.com>
import sys

sys.path.extend(['/Users/luoyouheng/Documents/nlp_research/nlp_research'])
from models.model import Model
from models.statistical_method.decision_tree.cart import Cart
import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
from sklearn.metrics import mean_squared_error, accuracy_score
from collections import Counter


class GBDT(Model):
    """实现梯度提升树算法，回归算法的损失函数为MSE"""

    def __init__(self, mode='regression', shrinkage=0.1, n_estimators=5, max_depth=3, min_samples=2, classes=None):
        """初始化参数
        : param mode : 模型的类别 1.'regression', 2.'binaryClassifier', 3.'multiClassifier'
        : param shrinkage : shrinkage 参数，每次缩减的大小
        : param n_estimators : 生成多少颗树
        : param max_depth : 最大深度
        : param min_samples : 划分包含最少数据的个数
        : param classes : 多分类时的参数，有多少类别
        : param init_value : 树的初始化值
        : param trees : 所有的树
        """
        self.mode = mode
        self.shrinkage = shrinkage
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.classes = classes
        self.init_value = None
        self.trees = {}

    def train(self, train_data, labels):
        """模型训练, 使用在决策树章节中实现的Cart树作为单个分类器
        : param train_data : 训练集，二维数组
        : param labels : 一维数组
        """
        row, col = train_data.shape
        if self.mode == 'regression':
            # 初始化，第一棵树的输出为所有树的均值, 更新残差
            self.init_value = np.mean(labels)
            labels = labels - self.init_value
            for num in range(1, self.n_estimators + 1):
                print('构建第 %d 棵树' % num)
                cur_tree = Cart(mode='regression', max_depth=self.max_depth, min_sample=self.min_samples)
                cur_tree.train(train_data, labels)
                self.trees[num] = cur_tree
                cur_labels = self.shrinkage * np.array(cur_tree.predict(train_data))
                labels -= cur_labels
        elif self.mode == 'binaryClassifier':
            # 初始化，第一棵树的输出为对数几率，通过求解导数为0的点。
            pi = np.sum(labels) / row
            self.init_value = np.log(pi / (1 - pi))
            pre = np.array([self.init_value] * row)
            residual = labels - self._sigmoid(pre)
            for num in range(1, self.n_estimators + 1):
                print('构建第 %d 棵树' % num)
                cur_tree = Cart(mode='regression', max_depth=self.max_depth, min_sample=self.min_samples)
                cur_tree.train(train_data, residual, labels, 'binary')
                self.trees[num] = cur_tree
                pre += self.shrinkage * np.array(cur_tree.predict(train_data))
                residual = labels - self._sigmoid(pre)
        elif self.mode == 'multiClassifier':
            # 初始化输出值 类别为0, 1, 2 ...
            self.init_value = {}
            for item in Counter(labels).most_common():
                self.init_value[item[0]] = item[1] / row
            # 将labels转化为每个类别的ont-hot编码
            one_hot_classes = np.zeros([self.classes, row])
            for i in range(self.classes):
                index = np.where(labels == i)
                cur = np.zeros(row)[index] = 1
                one_hot_classes[i] = cur
            pre = np.zeros([self.classes, row])
            for k, v in self.init_value.items():
                pre[k] = v
            # 计算residual
            residual = one_hot_classes - self._softmax(pre)
            for num in range(1, self.n_estimators):
                cur_estimator = {}
                for j in range(self.classes):
                    print('开始构建第 %d 轮的 %d 棵树' % (num, j))
                    cur_tree = Cart(mode='regression', max_depth=self.max_depth, min_sample=self.min_samples)
                    cur_tree.train(train_data, residual[j], one_hot_classes[j], 'multi')
                    cur_estimator[j] = cur_tree
                    pre[j] += np.array(cur_tree.predict(train_data))
                residual = one_hot_classes - self._softmax(pre)
                self.trees[num] = cur_estimator

    def _sigmoid(self, x):
        # 防溢出处理
        # if x < 0:
        # 	return np.exp(x) / ( np.exp(x) + 1)
        # else:
        # 	return 1 / (1 + np.exp(-x))
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """注意方向, axis=0"""
        exp = np.exp(x)
        return exp / np.sum(exp, axis=0)

    def predict(self, test_data):
        """预测结果
        :param test_data : 测试集，二维数组
        return : list
        """
        if self.mode != 'multiClassifier':
            labels = np.array([self.init_value] * test_data.shape[0])
            for i in range(1, self.n_estimators + 1):
                labels += self.shrinkage * np.array(self.trees[i].predict(test_data))
            if self.mode == 'regression':
                return labels
            elif self.mode == 'binaryClassifier':
                labels = np.array(labels)
                labels = self._sigmoid(labels)
                max_index = np.where(labels >= 0.5)
                min_index = np.where(labels < 0.5)
                labels[max_index] = 1
                labels[min_index] = 0
                return labels
        else:
            labels = np.zeros([self.classes, test_data.shape[0]])
            for k, v in self.init_value.items():
                labels[k] = v
            for num in range(1, self.n_estimators + 1):
                for j in range(self.classes):
                    labels[j] += self.shrinkage * self.trees[num][j].predict(test_data)
            return np.argmax(self._softmax(labels), axis=0)

    def dump(self, model_path):
        pass

    def load(self, model_path):
        pass


if __name__ == '__main__':
    # GBDT回归树
    # data = load_boston(return_X_y=True)
    # size = data[0].shape[0]
    # split = int(size * 0.8)
    # X_train, y_train = data[0][:split], data[1][:split]
    # X_test, y_test = data[0][split:], data[1][split:]
    # model = GBDT(mode='regression', shrinkage=0.2, n_estimators=12, max_depth=6, min_samples=2)
    # model.train(X_train, y_train)
    # pre = model.predict(X_test)
    # print(mean_squared_error(y_test, pre))

    # GBDT二分类树
    # data = load_breast_cancer(return_X_y=True)
    # size = data[0].shape[0]
    # split = int(size * 0.8)
    # X_train, y_train = data[0][:split], data[1][:split]
    # X_test, y_test = data[0][split:], data[1][split:]
    # model = GBDT(mode='binaryClassifier', shrinkage=0.1, n_estimators=5, max_depth=5, min_samples=3)
    # model.train(X_train, y_train)
    # pre = model.predict(X_test)
    # print(accuracy_score(y_test, pre))

    # GBDT多分类树
    data = load_iris(return_X_y=True)
    x = data[0]
    y = data[1]
    model = GBDT(mode='multiClassifier', shrinkage=0.1, n_estimators=8, max_depth=5, min_samples=3, classes=3)
    model.train(x, y)
    predict = model.predict(x)
    print(predict)
    # print(accuracy_score(data[1], predict))
