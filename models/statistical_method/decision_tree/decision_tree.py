# -*- encoding:utf-8 -*-
# @Time    : 2020/3/25 10:51 上午
# @Author  : HengGeZhiZou <1018676477@qq.com>
import sys
sys.path.extend(['/Users/luoyouheng/Documents/nlp_research/nlp_research'])
from models.model import Model
import numpy as np
from collections import Counter


class _TreeNode:
    """ID3和C4.5都是多叉树结构"""

    def __init__(self, val):
        # val代表用的第多少维的特征进行划分
        self.val = val
        self.level = None
        self.label = None
        self.children = []
        # 使用字典记录划分每个特征取值对应的子节点
        self.split_dict = dict()


class DecisionTree(Model):
    """实现决策树模型"""

    def __init__(self, mode='gain', threshold=0.02):
        """
        初始化模型
        :param mode: 构建决策树使用的划分方式
        1.gain: 使用信息增益的方式来划分决策树
        2.gain_ratio: 使用信息增益比的方式来划分
        3.gini :使用gini系数来划分
        :param threshold: 阈值
        """
        super().__init__()
        self.mode = mode
        self.threshold = threshold
        self.root = None

    def train(self, train_data, labels):
        """
        构建决策树
        :param train_data: 训练数据集，二维数组
        :param labels: 训练数据集的标签，一维数组
        """
        # 用来保存当前特征是否被使用过
        used = [False] * train_data.shape[1]

        def build_tree(data, label):
            row, col = data.shape
            # 树停止生长的条件 1.特征集为空 2.所有数据类别相同
            if sum(used) == col:
                node = _TreeNode(None)
                node.label = Counter(label).most_common()[0][0]
                return node
            elif len(set(label)) == 1:
                node = _TreeNode(None)
                node.label = label[0]
                return node
            # 计算切分点 1.当前节点的信息熵 2.每个特征的条件熵,计算之前先判断当前特征是否已经被使用
            cur_gain = []
            hd = self.cal_entropy(label)
            for i in range(col):
                if used[i] == True:
                    cur_gain.append(- float('inf'))
                    break
                else:
                    x = data[:, i]
                    cur_feature_ent = self.cal_condition_entropy(x, label)
                    # 选择模式
                    if self.mode == 'gain':
                        cur_gain.append(hd - cur_feature_ent)
                    elif self.mode == 'gain_ratio':
                        cur_gain.append((hd - cur_feature_ent) / self.cal_entropy(x))
            cur_fea_index = np.argmax(cur_gain)
            # 3.当信息增益小于阈值，树也停止生长
            if cur_gain[cur_fea_index] < self.threshold:
                node = _TreeNode(None)
                node.label = Counter(label).most_common()[0][0]
                return node
            cur_node = _TreeNode(cur_fea_index)
            used[cur_fea_index] = True
            # 按信息增益最大的特征来划分节点 并且去除划分的这个特征
            for i in set(data[:, cur_fea_index]):
                fea_index = np.where(data[:, cur_fea_index] == i)[0]
                res = build_tree(data[fea_index, :], label[fea_index])
                cur_node.split_dict[i] = res
                cur_node.children.append(res)
            return cur_node

        self.root = build_tree(train_data, labels)

    def cal_entropy(self, y):
        """计算信息熵"""
        size = y.shape[0]
        label_count = dict()
        for i in y:
            label_count[i] = label_count.get(i, 0) + 1
        ent = 0
        for value in label_count.values():
            ent += -1 * (value / size) * np.log2(value / size)
        return ent

    def cal_condition_entropy(self, x, y):
        """
        计算每一个特征的条件熵
        :param x: 一维数组，特征
        :param y: 一维数组，label
        :return: 当前特征的条件熵
        """
        size = x.shape[0]
        fea_expectation = dict()
        for i in x:
            fea_expectation[i] = fea_expectation.get(i, 0) + 1
        # 计算每种情况下的熵
        condition_ent = 0
        for key, value in fea_expectation.items():
            index = np.where(x == key)
            fea_ent = self.cal_entropy(y[index])
            condition_ent += (value / size) * fea_ent
        return condition_ent

    def predict(self, test_data):
        """
        :param test_data: 二维数组
        :return: 一维数组
        """
        label = []
        for i in test_data:
            cur = self.search(self.root, i)
            label.append(cur)
        return label

    def search(self, root, data):
        if not root.val:
            return root.label
        child = root.split_dict[data[root.val]]
        return self.search(child, data)

    def dump(self, **args):
        pass

    def load(self, **args):
        pass


if __name__ == '__main__':
    # 书中例子
    model = DecisionTree(mode='gain')
    x = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 1, 0, 1],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [1, 0, 1, 2],
                  [1, 0, 1, 2],
                  [2, 0, 1, 2],
                  [2, 0, 1, 1],
                  [2, 1, 0, 1],
                  [2, 1, 0, 2],
                  [2, 0, 0, 0]])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    model.train(x, y)
    print(model.predict([[2, 0, 0, 0]]))
