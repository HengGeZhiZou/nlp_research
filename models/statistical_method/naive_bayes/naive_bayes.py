# -*- encoding:utf-8 -*-
# @Time    : 2020/3/24 6:05 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from models.model import Model
import numpy as np
import os
import pickle


class NaiveBayes(Model):
    """实现朴素贝叶斯模型"""

    def __init__(self, _lambda=1):
        """
        :param _lambda: 平滑参数，等于一时为拉普拉斯平滑
        """
        super().__init__()
        self._lambda = _lambda
        self.prior = dict()
        self.condition_pro = dict()
        self.label_count = None

    def train(self, train_data, labels):
        """
        使用极大似然估计统计参数
        :param train_data: 训练数据，二维数组
        :param labels: 训练数据的标签
        """
        row, col = train_data.shape
        # 统计类别的先验概率
        label_dict = dict()
        for i in labels:
            label_dict[i] = label_dict.get(i, 0) + 1
        self.label_count = len(label_dict)
        for key, value in label_dict.items():
            self.prior[key] = (value + self._lambda) / (row + self.label_count * self._lambda)
        # 统计条件概率
        features_dict = dict()
        # 平滑统计
        features_count = [set()] * col
        for features, label in zip(train_data, labels):
            for i, f in enumerate(features):
                # 特征的纬度和取值
                cur = str(i + 1) + '_' + str(f)
                features_dict[(cur, label)] = features_dict.get((cur, label), 0) + 1
                features_count[i].add(f)

        for key, value in features_dict.items():
            cur_fea = int(key[0].split('_')[0]) - 1
            self.condition_pro[key] = (value + self._lambda) / (label_dict[key[1]] + len(features_count[cur_fea]))

    def predict(self, test_data):
        """
        预测结果
        :param test_data: 测试数据，二维数组
        :return:
        """
        label = []
        for row in test_data:
            label_res = dict()
            for key, value in self.prior.items():
                p_xy = value
                for i, f in enumerate(row):
                    cur = str(i + 1) + '_' + str(f)
                    cur_pro = self.condition_pro[(cur, key)]
                    p_xy *= cur_pro
                label_res[key] = p_xy
            label.append(max(label_res, key=label_res.get))
        return label

    def dump(self, model_path):
        """
        :param model_path: 保存模型地址
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = {'lambda': self._lambda, 'prior': self.prior, 'condition_pro': self.condition_pro,
                 'label_count': self.label_count}
        pickle.dump(model, open(model_path + '/model', mode='wb'))

    def load(self, model_path):
        """
        :param model_path: 恢复模型
        """
        model = pickle.load(open(model_path + '/model', mode='rb'))
        self._lambda = model['lambda']
        self.prior = model['prior']
        self.condition_pro = model['condition_pro']
        self.label_count = model['label_count']


if __name__ == '__main__':
    model = NaiveBayes()
    x = np.array(
        [[1, 1], [1, 2], [1, 2], [1, 1], [1, 1], [2, 1], [2, 2], [2, 2], [2, 3], [2, 3], [3, 3], [3, 2], [3, 2], [3, 3],
         [3, 3]])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    model.train(x, y)
    print(model.predict(np.array([[2, 1], [1, 2]])))
