# -*- encoding:utf-8 -*-
# @Time    : 2020/5/5 11:30 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRF(nn.Module):
    def __init__(self, num_labels):
        super(CRF, self).__init__()
        # 转移矩阵
        self.trans = torch.randn(num_labels, num_labels)
        self.num_labels = num_labels

    def sequence_score(self, inputs, labels):
        return self._crf_unary_score(inputs, labels) + self._crf_binary_score(inputs, labels)

    def crf_log_norm(self, inputs):
        # 每个batch分别计算
        scores = []
        for input in inputs:
            init_alphas = torch.full([1, self.num_labels], -10000)
            init_alphas[0][0] = 0
            init_alphas[0][3] = 0
            forward_var = init_alphas
            for item in input:
                alpha_t = []
                for tag in range(self.num_labels):
                    emit_score = item[tag].expand(1, self.num_labels)
                    trans_score = self.trans[:, tag].view(1, -1)
                    cur = forward_var + trans_score + emit_score
                    res = self._log_sum_exp(cur).view(1)
                    alpha_t.append(res)
                forward_var = torch.cat(alpha_t).view(1, -1)
            scores.append(self._log_sum_exp(forward_var).view(1))
        scores = torch.cat(scores).view(-1, 1)
        return scores

    def _log_sum_exp(self, vec):
        # 防溢出处理
        max_score = torch.max(vec)
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score)))

    def _crf_unary_score(self, inputs, tag_indices):
        # inputs 为上一层网络的输出大小[batch_size, seq_len, num_classes]
        # tag_indices 为标签的one-hot编码大小为[batch_size, seq_len, num_classes]
        score = inputs * tag_indices
        return torch.sum(torch.sum(score, 2), 1, keepdim=True)

    def _crf_binary_score(self, inputs, tag_indices):
        # pre shape:[batch_size, seq_len, num_labels, 1]
        # back shape:[batch_size, seq_len, 1, num_labels]
        pre = torch.unsqueeze(tag_indices[:, :-1], 3)
        back = torch.unsqueeze(tag_indices[:, 1:], 2)
        res = pre * back
        trans = torch.unsqueeze(torch.unsqueeze(self.trans, 0), 0)
        return torch.sum(torch.sum(trans * res, [2, 3]), 1, keepdim=True)

    def _viterbi(self, inputs):
        # inputs shape:[batch_size, seq_len, num_classes]
        batches_scores, batches_path = [], []
        for input in inputs:
            init_val = torch.full([1, self.num_labels], -10000.0)
            init_val[0][0] = 0
            init_val[0][3] = 0
            forward_var = init_val
            batch_road = []
            for item in input:
                cur_var = []
                road = []
                for i in range(self.num_labels):
                    trans_score = self.trans[:, i].view(1, -1)
                    cur = trans_score + forward_var
                    value, index = torch.max(cur, 1)
                    cur_var.append(value)
                    road.append(index.item())
                forward_var = (torch.cat(cur_var) + item).view(1, -1)
                batch_road.append(road)
            best_score, best_path_id = torch.max(forward_var, 1)
            best_path = []
            for item in reversed(batch_road):
                best_path_id = item[best_path_id]
                best_path.append(best_path_id)
            best_path.reverse()
            batches_scores.append(best_score)
            batches_path.append(best_path)
        return batches_scores, batches_path

    def forward(self, inputs):
        # inputs:[batch_size, seq_len, num_classes]
        scores, paths = self._viterbi(inputs)
        return inputs, scores, paths

    def get_loss(self, inputs, target):
        seq = self.sequence_score(inputs, target)
        z = self.crf_log_norm(inputs)
        loss = z - seq
        return torch.mean(loss)
