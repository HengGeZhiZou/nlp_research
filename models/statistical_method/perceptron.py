# -*- encoding:utf-8 -*-
# @Time    : 2020/3/23 5:10 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from models.model import Model
import numpy as np
import random
from tqdm import tqdm
import logging
from sklearn.datasets import load_breast_cancer
import os
import pickle


class Perceptron(Model):
    """实现感知机模型"""

    def __init__(self, max_iter=500, eta=0.0001):
        """
        初始化感知机的参数
        :param max_iter: 最大迭代数量
        :param eta: 学习率
        """
        super().__init__()
        self.max_iter = max_iter
        self.eta = eta
        self.w = 0
        self.b = 0

    def train(self, train_data, label):
        """
        :param train_data: 训练数据集，一个二维数组
        :param label: 训练集的标签，一维数组
        """
        row, col = train_data.shape
        self.w = np.zeros(col)
        # 开始迭代
        logger.info("开始迭代")
        for _ in tqdm(range(self.max_iter)):
            # 搜索所有负样本
            neg_sample = []
            for j in range(row):
                x = train_data[j]
                # 将类别为0的转化为-1
                y = 2 * label[j] - 1
                if -1 * y * (np.dot(self.w, x.T) + self.b) >= 0:
                    neg_sample.append(j)
            if not neg_sample:
                break
            cur = random.choice(neg_sample)
            cur_x = train_data[cur]
            cur_y = 2 * label[cur] - 1
            self.w = self.w + self.eta * cur_x * cur_y
            self.b = self.b + self.eta * cur_y
        logger.info('完成训练')

    def predict(self, test_data):
        """
        :param test_data: 测试数据，二维数组
        :return: 返回预测结果
        """
        result = np.dot(test_data, self.w.T) + self.b
        return np.maximum(np.sign(result), 0)

    def dump(self, model_path):
        """
        :param model_path: 保存模型的路径
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = {'w': self.w, 'b': self.b}
        pickle.dump(model, open(model_path + '/model', mode='wb'))

    def load(self, model_path):
        """
        :param model_path: 加载模型的路径
        """
        model = pickle.load(open(model_path + '/model', mode='rb'))
        self.w = model['w']
        self.b = model['b']


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    data = load_breast_cancer(return_X_y=True)
    size = data[0].shape[0]
    split = int(size * 0.8)
    X_train, y_train = data[0][:split], data[1][:split]
    X_test, y_test = data[0][split:], data[1][split:]
    model = Perceptron()
    model.train(X_train, y_train)
    prediction = model.predict(X_test)
    count = 0
    score = 0
    for i, j in zip(y_test, prediction):
        if i == j:
            score += 1
        count += 1
    print('准确率：%f' % (score / count))
