# -*- encoding:utf-8 -*-
# @Time    : 2020/5/6 11:11 上午
# @Author  : HengGeZhiZou <1018676477@qq.com>
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim

sys.path.extend(['/Users/luoyouheng/Documents/nlp_research/nlp_research/'])
from models.statistical_method.crf.data_util import Data
from models.statistical_method.crf.crf import CRF


class BILSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_labels, hidden_size):
        super(BILSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_size * 2, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, inputs):
        # inputs shape:[batch_size, seq_len]
        emb = self.embedding(inputs)
        output, _ = self.lstm(emb)
        out = self.hidden2tag(output)
        out, scores, paths = self.crf(out)
        return out, scores, paths

    def neg_log_likelihood(self, inputs, labels):
        labels = F.one_hot(labels, num_classes=self.num_labels)
        return self.crf.get_loss(inputs, labels)

    def cross_ent(self, inputs, labels):
        labels = F.one_hot(labels, num_classes=self.num_labels)
        soft = torch.softmax(inputs, 2)
        log_loss = -1 * labels * torch.log(soft)
        loss = torch.sum(log_loss, 2)
        return torch.mean(loss)


if __name__ == '__main__':
    data = Data(batch_size=32, max_seq_length=128)
    data.read_data('peoples_daily.txt')
    model = BILSTM_CRF(len(data.word2id), 200, 4, 150)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(10):
        print('============ 当前为第 %d 个epoch ============' % (epoch + 1))
        for num, (train, labels, seq_len) in enumerate(data):
            outputs, scores, paths = model(train)
            model.zero_grad()
            loss = model.neg_log_likelihood(outputs, labels)
            loss.backward()
            optimizer.step()
            # if num % 50 == 0:
            print('当前为第 %d 个batch, loss 为 %f' % (num, loss))
            total = (labels.shape[0] * labels.shape[1]) * 1.0
            paths = torch.tensor(paths)
            acc = (paths == labels).sum() / total
            print('当前准确率为 %f' % acc)
