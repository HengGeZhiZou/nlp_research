# -*- encoding:utf-8 -*-
# @Time    : 2020/4/8 11:42 上午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from models.model import Model
import numpy as np
from tqdm import tqdm
import os
import pickle


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

    def train(self, **args):
        pass

    def _viterbi(self, sentence):
        T = len(sentence)
        N = self.A.shape[0]
        weights = np.zeros(shape=(N, T))
        path = np.zeros(shape=(N, T))
        for t in range(T):
            if t == 0:
                weights[:, t] = self.pi + self.B[sentence[t]]
            else:
                for i in range(N):
                    cur = weights[:, t - 1] + \
                        self.A[:, i] + self.B[sentence[t]][i]
                    weights[i][t] = max(cur)
                    path[i][t] = int(np.argmax(cur))
        road = []
        road.append(np.argmax(weights[:, T - 1]))
        for r_t in reversed(range(T)):
            if r_t == 0:
                break
            last = road[-1]
            cur = path[last][r_t]
            road.append(int(cur))
        return road[::-1]

    def predict(self, test_data):
        """利用维特比算法解码
        :param test_data: 每条数据为一个句子
        """
        prediction = []
        for line in test_data:
            res = self._viterbi(line)
            cur = ''
            for i, j in zip(res, line):
                cur += j
                if i == 2 or i == 3:
                    cur += ' '
            prediction.append(cur)
        return prediction

    def dump(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = {'pi': self.pi, 'A': self.A, 'B': self.B}
        pickle.dump(model, open(model_path + '/model', mode='wb'))

    def load(self, model_path):
        model = pickle.load(open(model_path, mode='rb'))
        self.pi = model['pi']
        self.A = model['A']
        self.B = model['B']


def mle_parameters(path, labels):
    """利用极大似然估计参数, 使用 BMES 标注"""
    size = len(labels)
    pi = np.zeros([size])
    A = np.zeros([size, size])
    B = {}
    train_data = [s.split()
                  for s in open(path, mode='r', encoding='utf-8').readlines()]
    for line in tqdm(train_data):
        pre = ''
        for num, word in enumerate(line):
            # 起始词
            word_len = len(word)
            if num == 0 and word != '':
                if word_len == 1:
                    pi[3] += 1
                else:
                    pi[0] += 1
            if word_len == 1:
                if word not in B:
                    B[word] = np.zeros(shape=size)
                B[word][3] += 1
                if pre == 'E':
                    A[2][3] += 1
                elif pre == 'S':
                    A[3][3] += 1
                pre = 'S'
            elif word_len > 1:
                for n, char in enumerate(word):
                    if char not in B:
                        B[char] = np.zeros(shape=size)
                    if n == 0:
                        B[char][0] += 1
                        if pre == 'E':
                            A[2][0] += 1
                        elif pre == 'S':
                            A[3][0] += 1
                        pre = 'B'
                    elif n == word_len - 1:
                        B[char][2] += 1
                        if pre == 'B':
                            A[0][2] += 1
                        elif pre == 'M':
                            A[1][2] += 1
                        pre = 'E'
                    else:
                        B[char][1] += 1
                        if pre == 'B':
                            A[0][1] += 1
                        elif pre == 'M':
                            A[1][1] += 1
                        pre = 'M'
    # 转化为概率
    pi_sum = sum(pi)
    for i in range(pi.shape[0]):
        if pi[i] == 0:
            pi[i] = -3.14e+100
        else:
            pi[i] = np.log(pi[i] / pi_sum)
    for hidden in A:
        h_sum = sum(hidden)
        for i in range(hidden.shape[0]):
            if hidden[i] == 0:
                hidden[i] = -3.14e+100
            else:
                hidden[i] = np.log(hidden[i] / h_sum)
    b_sum = np.zeros([size])
    for _, v in B.items():
        b_sum += v
    for _, v in B.items():
        for i in range(size):
            if v[i] == 0:
                v[i] = -3.14e+100
            else:
                v[i] = np.log(v[i] / b_sum[i])
    return pi, A, B


if __name__ == '__main__':
    # 书中实例
    pi = np.array([0.2, 0.4, 0.4])
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = {'红': [0.5, 0.4, 0.7], '白': [0.5, 0.6, 0.3]}
    model = Hmm(pi, A, B)
    print(model.cal_probability(['红', '白', '红'], mode='backward'))

    # 使用监督信息训练一个HMM分词模型
    labels = ['B', 'M', 'E', 'S']
    pi, A, B = mle_parameters(
        '/models/statistical_method/hmm/peoples_daily.txt', labels=labels)
    model = Hmm(pi, A, B)
    print(model.predict(['王光美在陕西大荔县看望贫困母亲', '她叫黄继美，住在四川宣汉县落耳坡村。']))
