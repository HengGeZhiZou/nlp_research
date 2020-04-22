# -*- encoding:utf-8 -*-
# @Time    : 2020/4/2 11:43 上午
# @Author  : HengGeZhiZou <1018676477@qq.com>
import sys
sys.path.extend(['/Users/luoyouheng/Documents/nlp_research/nlp_research'])
from models.model import Model
import numpy as np


class BasicClassifier():
    """基分类器"""

    def __init__(self, weights, step):
        self.weights = weights
        self.step = step
        self.min_error = float('inf')
        self.feature_index = None
        self.feature_value = None
        self.label = None
        self.label_dir = None
        self.alpha = None

    def train(self, train_data, labels):
        row, col = train_data.shape
        for i in range(col):
            cur_data = train_data[:, i]
            min_value = min(cur_data)
            max_value = max(cur_data)
            # 得到每个划分点
            for j in range(min_value + self.step, max_value + self.step, self.step):
                # 分两种情况：1.小于划分点的为正例，2.大于划分点的为正例
                pos_split = [1 if n < j else -1 for n in cur_data]
                neg_split = [-1 if n < j else 1 for n in cur_data]
                # 分别统计误差
                pos_error = 0.0
                neg_error = 0.0
                for p, n, l, w in zip(pos_split, neg_split, labels, self.weights):
                    if p != l:
                        pos_error += w
                    if n != l:
                        neg_error += w
                if pos_error < neg_error:
                    cur_error = pos_error
                    cur_label = pos_split
                    cur_label_dir = 'pos'
                else:
                    cur_error = neg_error
                    cur_label = neg_split
                    cur_label_dir = 'neg'
                if self.min_error > cur_error:
                    self.min_error = cur_error
                    self.feature_value = j
                    self.feature_index = i
                    self.label = cur_label
                    self.label_dir = cur_label_dir
        self.alpha = 0.5 * np.log((1 - self.min_error) / self.min_error)
        weights = np.zeros(row)
        for num, i in enumerate(self.weights):
            weights[num] = self.weights[num] * np.exp(-1 * self.alpha * labels[num] * self.label[num])
        weights = weights / np.sum(weights)
        return weights

    def predict(self, test_data):
        cur_test = test_data[:, self.feature_index]
        test_index_left = np.where(cur_test < self.feature_value)[0]
        if self.label_dir == 'pos':
            label = - np.ones(test_data.shape[0])
            label[test_index_left] = 1
        else:
            label = np.ones(test_data.shape[0])
            label[test_index_left] = -1
        return label * self.alpha


class Adaboost(Model):
    """实现adaboost模型"""

    def __init__(self, max_iter=100, step=1):
        super().__init__()
        self.max_iter = max_iter
        self.weights = None
        # 每次增加的步数
        self.step = step
        # 存储多个分类器
        self.classifier = dict()
        self.classifier_nums = 0

    def train(self, train_data, labels):
        """
        :param train_data: 训练集，二维数组
        :param labels: 标签，一维数组
        """
        row, col = train_data.shape
        self.weights = [1.0 / row] * row
        for epoch in range(self.max_iter):
            print('当前是第 %d 个 epoch' % (epoch + 1))
            # 创建一个基分类器保存
            bc = BasicClassifier(self.weights, self.step)
            self.classifier_nums += 1
            self.classifier[epoch + 1] = bc
            self.weights = bc.train(train_data, labels)
            # 是否提前停止
            cur_label = self.predict(train_data)
            error = sum([1 for i in range(row) if cur_label[i] != labels[i]])
            if error == 0:
                break

    def predict(self, test_data):
        # 将每个分类器的结果相加，然后使用sign函数
        label = np.zeros(test_data.shape[0])
        for num in range(1, self.classifier_nums + 1):
            bc = self.classifier[num]
            predict = bc.predict(test_data)
            label += predict
        return np.sign(label)

    def dump(self, **args):
        pass

    def load(self, **args):
        pass


if __name__ == '__main__':
    #   书中例子
    train_data = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    labels = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    model = Adaboost()
    model.train(train_data, labels)
    print(model.classifier[3].alpha)
