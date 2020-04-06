# -*- encoding:utf-8 -*-
# @Time    : 2020/3/26 5:19 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from sklearn.model_selection import train_test_split

from models.model import Model
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import pickle


class LogisticRegression(Model):
    """逻辑回归模型"""

    def __init__(self, lr=0.01, max_iter=500, mode='multi', classes=3):
        """
        :param lr: 学习率
        :param max_iter: 迭代次数
        :param mode: 可以选择多分类或二分类 1.multi 2.binary
        """
        super().__init__()
        self.lr = lr
        self.max_iter = max_iter
        self.mode = mode
        self.classes = classes
        self.w = None
        self.b = None

    def train(self, train_data, labels):
        """
        训练逻辑回归模型,使用梯度上升算法训练
        :param train_data: 训练数据，二维数组
        :param labels: 训练集标签, 二分类时，需要一维的数组。多分类时，二维的one-hot数组
        """
        row, col = train_data.shape
        # 初始化参数
        if self.mode == 'binary':
            self.w = np.random.rand(col)
            self.b = 0
        else:
            self.w = np.random.rand(col, self.classes)
            self.b = np.zeros(self.classes)
        # 计算出逻辑回归的梯度为 x * (y - f(x))
        for epoch in range(self.max_iter):
            print('当前训练: %d epoch ' % epoch)
            x = np.matmul(train_data, self.w) + self.b
            if self.mode == 'binary':
                y = self._sigmoid(x)
                error = np.expand_dims(labels - y, axis=1)
                self.w += np.mean(train_data * error * self.lr, axis=0)
                self.b += np.mean(error)
            elif self.mode == 'multi':
                y = self._softmax(x)
                log_loss = self.cross_entropy(y, labels)
                print(log_loss)
                error = labels - y
                # 根据每一条样本来更新参数
                cur_w = np.zeros(shape=(col, self.classes))
                cur_b = np.zeros(self.classes)
                for i, j in zip(train_data, error):
                    res = np.dot(np.expand_dims(i, axis=1), np.expand_dims(j, axis=0))
                    cur_w += res * self.lr
                    cur_b += j
                self.w += (cur_w / row)
                self.b += (cur_b / row)

    def cross_entropy(self, y, label):
        """计算交叉熵 log loss"""
        return -1 * np.mean(np.sum(np.log2(y) * label, axis=1))

    def _sigmoid(self, x):
        """防止溢出"""
        y = []
        for i in x:
            if i > 0:
                y.append(1 / (1 + np.exp(-i)))
            else:
                y.append(np.exp(i) / (1 + np.exp(i)))
        return y

    def _softmax(self, x):
        """防止溢出"""
        m = np.max(x, axis=1, keepdims=True)
        pro = np.exp(x - m) / np.sum(np.exp(x - m), axis=1, keepdims=True)
        return pro

    def predict(self, test_data):
        # pro = np.array(self._sigmoid(np.matmul(test_data, self.w) + self.b))
        pro = self._softmax(np.matmul(test_data, self.w) + self.b)
        return np.argmax(pro, axis=1)

    def dump(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = {'w': self.w, 'b': self.b, 'lr': self.lr}
        pickle.dump(model, open(model_path + '/model', mode='wb'))

    def load(self, model_path):
        model = pickle.load(open(model_path + '/model', mode='rb'))
        self.w = model['w']
        self.b = model['b']
        self.lr = model['lr']


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:, :2], data[:, -1]


if __name__ == '__main__':
    # model = LogisticRegression(mode='binary')
    # data = load_breast_cancer(return_X_y=True)
    # size = data[0].shape[0]
    # split = int(size * 0.8)
    # X_train, y_train = data[0][:split], data[1][:split]
    # X_test, y_test = data[0][split:], data[1][split:]
    # model.train(X_train, y_train)
    # pred = model.predict(X_test)
    # print(accuracy_score(y_test, pred))
    model = LogisticRegression(mode='multi', classes=3)
    data = load_iris(return_X_y=True)
    x = data[0]
    y = np.eye(3)[data[1]]
    model.train(x, y)
    predict = model.predict(x)
    print(accuracy_score(data[1], predict))
    # print(model.w)
