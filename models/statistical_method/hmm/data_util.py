# -*- encoding:utf-8 -*-
# @Time    : 2020/4/12 4:05 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
import numpy as np
from tqdm import tqdm


def mle_parameters(path, labels_dict):
    """利用极大似然估计参数, 使用 BMES 标注"""
    size = len(labels_dict)
    pi = np.zeros([size])
    A = np.zeros([size, size])
    B = {}
    train_data = [s.split() for s in open(path, mode='r', encoding='utf-8').readlines()]
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
