# -*- encoding:utf-8 -*-
# @Time    : 2020/4/13 6:14 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys

sys.path.extend(['/Users/luoyouheng/Documents/nlp_research/nlp_research/'])
from models.statistical_method.crf.data_util import Data
from models.statistical_method.crf.crf import CRF


class Model(nn.Module):

    def __init__(self, vocab_size, embedding_size, filter_size, out_channel, dropout, num_classes):
        super(Model, self).__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        self.conv1 = nn.Conv1d(embedding_size, out_channel, kernel_size=filter_size, padding=1)
        self.conv2 = nn.Conv1d(embedding_size, out_channel, kernel_size=filter_size, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_size, num_classes)
        self.crf = CRF(num_classes)

    def forward(self, inputs):
        # 输入数据大小 [batch_size, seq_len]
        emb = self.embedding(inputs)
        # 通过词嵌入 [batch_size, seq_len, embedding_size]
        out = emb.permute(0, 2, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.crf(out)
        return out

    def cross_ent(self, pred, labels, num_classes):
        # pred shape:[batch_size, seq_len, hidden_size]
        # labels shape[batch_size, seq_len]
        label = F.one_hot(labels, num_classes=num_classes)
        return self.crf.get_loss(pred, label)


if __name__ == '__main__':
    data = Data(batch_size=32, max_seq_length=128)
    data.read_data('peoples_daily.txt')
    model = Model(len(data.word2id), 200, 3, 200, 0.3, 4)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(10):
        print('当前为第 %d 个epoch' % (epoch + 1))
        for num, (train, labels, seq_len) in enumerate(data):
            print('当前为第 %d 个batch' % (num + 1))
            outputs = model(train)
            model.zero_grad()
            loss = model.cross_ent(outputs, labels, 4)
            loss.backward()
            optimizer.step()
            print('loss 为 %f' % loss)
