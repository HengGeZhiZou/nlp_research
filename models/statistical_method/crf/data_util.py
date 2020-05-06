# -*- encoding:utf-8 -*-
# @Time    : 2020/5/5 5:50 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
import torch
import numpy as np
import os
from tqdm import tqdm
from collections import Iterable


class Data(object):

    def __init__(self, batch_size=32, max_seq_length=128):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.label2id = {'B': 0, 'I': 1, 'E': 2, 'O': 3}
        self.word2id = {}
        self.id2word = {}
        self.data = []
        self.labels = []
        self.seq_len = []
        self.index = 0
        self.batches = 0

    def read_data(self, file_path):
        data = [s.split()
                for s in open(file_path, mode='r', encoding='utf-8').readlines()]
        # 构建字典
        self.word2id = {'unk': 0, 'pad': 1}
        self.id2word = {0: 'unk', 1: 'pad'}
        count = 2
        # 标注结果
        labels = []
        for line in tqdm(data):
            label = []
            for word in line:
                if len(word) == 1:
                    label.append('O')
                else:
                    label.append('B')
                    for _ in range(len(word) - 2):
                        label.append('I')
                    label.append('E')
                for char in word:
                    if char not in self.word2id:
                        self.word2id[char] = count
                        self.id2word[count] = char
                        count += 1
            labels.append(label)
        # 转化结果
        data2id = []
        labels2id = []
        seq_len = []
        for line, line_label in zip(data, labels):
            cur_data = []
            cur_label = []
            for word in line:
                for char in word:
                    cur_data.append(self.word2id[char])
            for l in line_label:
                cur_label.append(self.label2id[l])
            if len(cur_data) > self.max_seq_length:
                cur_data = cur_data[:self.max_seq_length]
                cur_label = cur_label[:self.max_seq_length]
                seq_len.append(self.max_seq_length)
            else:
                seq_len.append(len(cur_data))
                cur_data += [1] * (self.max_seq_length - len(cur_data))
                cur_label += [3] * (self.max_seq_length - len(cur_label))
            data2id.append(cur_data)
            labels2id.append(cur_label)
        self.data = data2id
        self.labels = labels2id
        self.batches = len(self.data) // self.batch_size
        self.seq_len = seq_len

    def __next__(self):
        if self.index < self.batches:
            cur_data = self.data[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            cur_label = self.labels[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            seq_len = self.seq_len[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            cur_data = torch.LongTensor(cur_data)
            cur_label = torch.LongTensor(cur_label)
            seq_len = torch.LongTensor(seq_len)
            return (cur_data, cur_label, seq_len)
        else:
            self.index = 0
            raise StopIteration

    def __iter__(self):
        return self


if __name__ == '__main__':
    data = Data()
    data.read_data('peoples_daily.txt')
    for item in data:
        print(item[0])
        print(item[1])
        print(item[2])
