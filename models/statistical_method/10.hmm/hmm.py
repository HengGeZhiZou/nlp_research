# -*- encoding:utf-8 -*-
# @Time    : 2020/4/8 11:42 上午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from models.model import Model
import numpy as np


class Hmm(Model):
    """实现HMM模型，分为三个问题 1.概率计算问题 2.学习问题 3.解码问题"""

    def __init__(self, pi=None, A=None, B=None):
        super().__init__()
        # 初始概率，状态转移概率，观测概率
        self.pi = pi
        self.A = A
        self.B = B

    def _forward(self, inputs):
        """实现前向算法，返回每个时间步计算的alpha值"""
        T = len(inputs)
        N = self.A.shape[0]
        alpha = np.zeros(shape=[N, T])
        for t in range(T):
            if t == 0:
                alpha[:, t] = np.multiply(self.pi, self.B[inputs[t]])
            else:
                cur = np.dot(alpha[:, t - 1], self.A)
                alpha[:, t] = np.multiply(cur, self.B[inputs[t]])
        return alpha

    def _backward(self, inputs):
        """实现后向算法，返回每个时间步计算的beta值"""
        T = len(inputs)
        N = self.A.shape[0]
        beta = np.zeros(shape=[N, T])
        for t in reversed(range(T)):
            if t == T - 1:
                beta[:, t] = np.ones(N)
            else:
                cur = np.multiply(beta[:, t + 1], self.B[inputs[t + 1]])
                beta[:, t] = np.dot(cur, self.A.transpose())
        return beta

    def cal_probability(self, inputs, mode='forward'):
        """
        完成 1. 概率计算问题
        :param inputs: 观测结果，计算出现的概率
        :param mode: 选择前向和后向计算方法
        在计算前向和后向概率的过程中，注意乘法顺序
        """
        if mode == 'forward':
            alpha = self._forward(inputs)
            return sum(alpha[:, -1])
        elif mode == 'backward':
            beta = self._backward(inputs)
            res = np.multiply(beta[:, 0], self.B[inputs[0]])
            res = np.multiply(res, self.pi)
            return sum(res)

    def train(self, inputs, mode='EM'):
        """实现Hmm模型的训练
        :param inputs: 观测数据，一个二维矩阵
        :param mode: EM：无监督学习，使用EM迭代训练参数
        """
        pass

    def predict(self, **args):
        pass

    def dump(self, **args):
        pass

    def load(self, **args):
        pass


if __name__ == '__main__':
    # 书中实例
    pi = np.array([0.2, 0.4, 0.4])
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = {'红': [0.5, 0.4, 0.7], '白': [0.5, 0.6, 0.3]}
    model = Hmm(pi, A, B)
    print(model.cal_probability(['红', '白', '红'], mode='backward'))
