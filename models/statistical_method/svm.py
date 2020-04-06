# -*- encoding:utf-8 -*-
# @Time    : 2020/3/29 10:51 上午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from models.model import Model
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SVM(Model):

    def __init__(self, C=1.0, max_iter=100):
        super().__init__()
        # 拉格朗日乘子
        self.alpha = None
        self.b = None
        # 松弛变量
        self.C = C
        # 保存真实值和预测值之间的误差
        self.E = None
        self.max_iter = max_iter

    def train(self, train_data, labels):
        """实现 svm 的训练，利用SMO算法"""
        row, col = train_data.shape
        self.alpha = np.zeros(row)
        self.b = 0
        self.x = train_data
        self.y = labels
        # 先计算所有的E
        self.E = []
        for i in range(row):
            self.E.append(self._g(i) - self.y[i])
        for i in range(self.max_iter):
            print('当前训练 %d epoch' % i)
            # 选取两个需要更新的 alpha1, alpha2, 得到是 alpha 的下标
            a1_index, a2_index = self._get_alpha()
            a1_old, a2_old = self.alpha[a1_index], self.alpha[a2_index]
            x1, x2 = self.x[a1_index], self.x[a2_index]
            y1, y2 = self.y[a1_index], self.y[a2_index]
            # 计算 alpha2 剪辑后的解,计算剪辑范围
            if self.y[a1_index] != self.y[a2_index]:
                L = max(0, a2_old - a1_old)
                H = min(self.C, self.C + a2_old - a1_old)
            else:
                L = max(0, a2_old + a1_old - self.C)
                H = min(self.C, a2_old + a1_old)
            if L == H:
                continue
            eta = np.dot(x1, x1) + np.dot(x2, x2) - 2 * np.dot(x1, x2)
            a2_new_unc = a2_old + (self.y[a2_index] * (self.E[a1_index] - self.E[a2_index]) / eta)
            if a2_new_unc > H:
                a2_new = H
            elif L <= a2_new_unc <= H:
                a2_new = a2_new_unc
            else:
                a2_new = L
            a1_new = a1_old + y1 * y2 * (a2_old - a2_new)
            b1_new = self.b - self.E[a1_index] - self.y[a1_index] * np.dot(x1, x1) * (a1_new - a1_old) - self.y[
                a2_index] * np.dot(x2, x1) * (a2_new - a2_old)
            b2_new = self.b - self.E[a2_index] - self.y[a1_index] * np.dot(x1, x2) * (a1_new - a1_old) - self.y[
                a2_index] * np.dot(x2, x2) * (a2_new - a2_old)
            if 0 < a1_new < self.C:
                b_new = b1_new
            elif 0 < a2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2
            # 更新参数
            self.alpha[a1_index] = a1_new
            self.alpha[a2_index] = a2_new
            self.b = b_new
            self.E[a1_index] = self._E(a1_index)
            self.E[a2_index] = self._E(a2_index)

    def _get_alpha(self):
        size = len(self.alpha)
        index = []
        back_index = []
        # 外层循环
        for i in range(size):
            # 首先查找在支持向量上的点
            if 0 < self.alpha[i] < self.C:
                # 判断是否满足KKT条件
                if not self._KKT(i):
                    index.append(i)
            else:
                if not self._KKT(i):
                    back_index.append(i)
        # 找到不满足KKT条件的点
        index += back_index
        # 内层循环
        for i in index:
            E1 = self.E[i]
            if E1 >= 0:
                j = np.argmin(self.E)
            else:
                j = np.argmax(self.E)
            if j == i:
                continue
            else:
                return i, j

    def _KKT(self, i):
        """判断是否满足KKT条件"""
        g = self._g(i)
        if self.alpha[i] == 0 and g * self.y[i] >= 1:
            return True
        elif 0 < self.alpha[i] < self.C and g * self.y[i] == 1:
            return True
        elif self.alpha[i] == self.C and g * self.y[i] <= 1:
            return True
        return False

    def _g(self, i):
        """用来计算g（x）函数间隔,传入数据的下标"""
        r = self.b
        for j in range(self.x.shape[0]):
            r += self.alpha[j] * self.y[j] * np.dot(self.x[i], self.x[j])
        return r

    def _E(self, i):
        """计算误差"""
        return self._g(i) - self.y[i]

    def predict(self, test_data):
        label = []
        for i in test_data:
            res = 0
            for j in range(self.x.shape[0]):
                res += self.alpha[j] * self.y[j] * np.dot(i, self.x[j])
            res += self.b
            label.append(np.sign(res))
        return label

    def dump(self, **args):
        pass

    def load(self, **args):
        pass


def trans(y):
    labels = []
    for i in y:
        labels.append(2 * i - 1)
    return np.array(labels)


if __name__ == '__main__':
    model = SVM()
    data = load_breast_cancer(return_X_y=True)
    size = data[0].shape[0]
    split = int(size * 0.8)
    X_train, y_train = data[0][:split], data[1][:split]
    X_test, y_test = data[0][split:], data[1][split:]
    y_train = trans(y_train)
    y_test = trans(y_test)
    model.train(X_train, y_train)
    pred = model.predict(X_test)
    print(model.alpha)
    print(pred)
    print(accuracy_score(y_test, pred))
