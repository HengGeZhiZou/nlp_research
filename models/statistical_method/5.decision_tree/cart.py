# -*- encoding:utf-8 -*-
# @Time    : 2020/3/26 10:03 上午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from models.model import Model
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from collections import Counter
import os
import pickle


# cart 树为二叉树
class _TreeNode:
    def __init__(self, val):
        self.val = val
        self.split = None
        self.label = None
        self.left = None
        self.right = None


class Cart(Model):
    """实现cart树算法"""

    def __init__(self, mode='classifier', min_sample=2, the=0.05):
        """
        :param mode: 选择分类树或回归树 1. 'classifier 2.regression
        :param min_sample: 停止条件，每个叶节点最少包含节点
        :param the: 阈值，停止条件
        """
        super().__init__()
        self.mode = mode
        self.min_sample = min_sample
        self.the = the
        self.root = None

    def train(self, train_data, labels):
        """
        :param train_data: 训练数据集，二维数组
        :param labels: 训练集标签，一维数组
        """
        if self.mode == 'regression':
            self.build_regression(train_data, labels)
        elif self.mode == 'classifier':
            self.build_classifier(train_data, labels)

    def build_regression(self, train_data, labels):
        """构建cart回归树"""

        def build(data, label):
            row, col = data.shape
            # 构建cart回归树的停止条件 1.节点中样本个数小于小于阈值 2.样本基尼系数小于阈值（和ID3，C4.5不同，特征可以复用）
            if data.shape[0] <= self.min_sample:
                cur = _TreeNode(None)
                cur.label = np.mean(label)
                return cur
            min_error = float('inf')
            min_error_index = 0
            min_error_split = 0
            # 计算每个特征当前
            for features in range(col):
                curr_data = np.sort(data[:, features])
                for i in range(row - 1):
                    left_data, left_label = data[:i + 1], label[:i + 1]
                    right_data, right_label = data[i + 1:], label[i + 1:]
                    left_error = np.sum(np.square(left_label - np.mean(left_label)))
                    right_error = np.sum(np.square(right_label - np.mean(right_label)))
                    total_error = left_error + right_error
                    if total_error < min_error:
                        min_error = total_error
                        min_error_split = curr_data[i]
                        min_error_index = features
            if min_error < self.the:
                cur = _TreeNode(None)
                cur.label = np.mean(label)
                return cur
            cur_node = _TreeNode(min_error_index)
            cur_node.split = min_error_split
            left_index = np.where(data[:, min_error_index] <= min_error_split)[0]
            right_index = np.where(data[:, min_error_index] > min_error_split)[0]
            cur_node.left = build(data[left_index], label[left_index])
            cur_node.right = build(data[right_index], label[right_index])
            return cur_node

        self.root = build(train_data, labels)

    def build_classifier(self, train_data, labels):
        """
        利用gini系数构建分类二叉树
        :param train_data: 二维数组
        :param labels: 一维数组
        """
        used = [False] * train_data.shape[1]

        def build(data, label, used):
            row, col = data.shape
            # 构建cart分类树的停止条件，1. 节点包含样本个数小于阈值，2.节点gini指数小于阈值，3.所有节点属于一个类别,4.特征用完
            if row < self.min_sample or sum(used) == col or len(set(label)) == 1:
                cur = _TreeNode(None)
                cur.label = Counter(label).most_common()[0][0]
                return cur
            else:
                min_gini = float('inf')
                min_split_fea = 0
                min_split_value = 0
                min_index_y, min_index_n = 0, 0
                for features in range(col):
                    if used[features]:
                        continue
                    else:
                        for f, n in Counter(data[:, features]).most_common():
                            index_y = np.where(data[:, features] == f)[0]
                            index_n = np.where(data[:, features] != f)[0]
                            cur_gini = self._gini(label[index_y]) * (n / row) + self._gini(label[index_n]) * (
                                    (row - n) / row)
                            if cur_gini < min_gini:
                                min_gini = cur_gini
                                min_split_fea = features
                                min_split_value = f
                                min_index_y = index_y
                                min_index_n = index_n
                # 当出现完美划分的情况时（左右子树里只有一个类别），此时最小gini系数等于0。
                # if min_gini < self.the:
                #     cur = _TreeNode(None)
                #     cur.label = Counter(label).most_common()[0][0]
                #     return cur
                # else:
                used[min_split_fea] = True
                cur_node = _TreeNode(min_split_fea)
                cur_node.split = min_split_value
                cur_node.left = build(data[min_index_y], label[min_index_y], used)
                cur_node.right = build(data[min_index_n], label[min_index_n], used)
                return cur_node

        self.root = build(train_data, labels, used)

    def _gini(self, x):
        """计算gini系数"""
        size = x.shape[0]
        gini = 0
        for item in Counter(x).most_common():
            gini += (item[1] / size) * (1 - item[1] / size)
        return gini

    def predict(self, test_data):
        label = []
        for i in test_data:
            if self.mode == 'regression':
                res = self._regression_search(self.root, i)
            elif self.mode == 'classifier':
                res = self._classifier_search(self.root, i)
            label.append(res)
        return label

    def _regression_search(self, root, data):
        if root.label:
            return root.label
        if data[root.val] <= root.split:
            return self._regression_search(root.left, data)
        else:
            return self._regression_search(root.right, data)

    def _classifier_search(self, root, data):
        if root.label:
            return root.label
        if data[root.val] == root.split:
            return self._classifier_search(root.left, data)
        else:
            return self._classifier_search(root.right, data)

    def dump(self, model_path):
        """序列化保存树模型"""
        if os.path.exists(model_path):
            os.makedirs(model_path)
        tree_serialize = self._serialize(self.root, '')
        model = {'tree': tree_serialize, 'mode': self.mode, 'min_sample': self.min_sample, 'the': self.the}
        pickle.dump(model, open(model_path+'/model', mode='wb'))

    def _serialize(self, root, res):
        """ 序列化二叉树"""
        if not root:
            res += 'None,'
        else:
            res += (str(root.val) + '|' + str(root.label) + '|' + str(root.split) + ',')
            res = self._serialize(root.left, res)
            res = self._serialize(root.right, res)
        return res

    def _deserialize(self, data):
        """反序列化二叉树"""
        if data[0] == 'None':
            data.pop(0)
            return None
        else:
            cur = data.pop(0)
            root = _TreeNode(None)
            val, label, split = cur.split('|')
            root.val = int(val)
            root.label = int(label)
            root.split = int(split)
            root.left = self._deserialize(data)
            root.right = self._deserialize(data)
            return root

    def load(self, model_path):
        model = pickle.load(open(model_path + '/model', mode='rb'))
        self.mode = model['mode']
        self.min_sample = model['min_sample']
        tree_serialize = model['tree']
        self.the = model['the']
        tree_list = tree_serialize.split(',')
        self.root = self._deserialize(tree_list)


if __name__ == '__main__':
    # 回归树
    model = Cart(mode='regression')
    data = load_boston(return_X_y=True)
    size = data[0].shape[0]
    split = int(size * 0.8)
    X_train, y_train = data[0][:split], data[1][:split]
    X_test, y_test = data[0][split:], data[1][split:]
    model.train(X_train, y_train)
    pre = model.predict(X_test)
    print(mean_squared_error(y_test, pre))

    # 分类树
    model = Cart(mode='classifier')
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
    print(model.predict([[1, 1, 1, 2]]))
