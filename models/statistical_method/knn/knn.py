# -*- encoding:utf-8 -*-
# @Time    : 2020/3/23 10:01 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
import sys
sys.path.extend(['/Users/luoyouheng/Documents/nlp_research/nlp_research'])
from models.model import Model
import numpy as np
import logging
import heapq
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import os
import pickle


class _TreeNode:
    """构造 kd 树的节点"""

    def __init__(self, val, label=None, split=None, left=None, right=None):
        self.val = val
        self.label = label
        self.split = split
        self.left = left
        self.right = right


class KNN(Model):
    """实现k近邻模型"""

    def __init__(self, k=None, p=None):
        """
        knn中的参数
        :param k: 最近邻点的个数
        :param p: 计算距离的方式
        """
        super().__init__()
        self.k = k
        self.p = p
        self.root = None

    def train(self, train_data, label):
        """
        knn 没有显示的训练过程，训练过程为构建二叉树
        :param train_data: 训练数据集，二维数组
        :param label: 训练集标签，一维数组
        """

        # 递归构建二叉树
        logger.info("开始构建kd树")

        def build_tree(data, label, depth):
            row, col = data.shape
            if row == 0: return None
            # 选择切分特征
            cur_split = depth % col
            data = data[np.argsort(data[:, cur_split])]
            label = label[np.argsort(data[:, cur_split])]
            mid = (row - 1) // 2
            root = _TreeNode(data[mid])
            root.label = label[mid]
            root.split = cur_split
            root.left = build_tree(data[:mid, :], label[:mid], depth + 1)
            root.right = build_tree(data[mid + 1:, :], label[mid + 1:], depth + 1)
            return root

        self.root = build_tree(train_data, label, 0)
        logger.info("构建完成")

    def predict(self, test_data):
        """
        :param test_data: 测试数据，二维矩阵
        :return: 二维矩阵
        """
        row, col = test_data.shape
        predictions = []
        for i in range(row):
            cur = test_data[i]
            cur_res = self._search(_TreeNode(cur))
            cur_label = [j[1].label for j in cur_res]
            predictions.append(Counter(cur_label).most_common(1)[0][0])
        return predictions

    def _search(self, node):
        """
        查找最近的 k 个点
        :param node: 待查找节点
        :return: 最近k个点
        """
        # 使用小根堆来保存距离
        top_k_dis = [(-float('inf'), None)] * self.k

        def visit(root):
            if not root: return
            if root.val[root.split] > node.val[root.split]:
                visit(root.left)
            else:
                visit(root.right)
            node_dis = self._distance(root.val, node.val)
            heapq.heappushpop(top_k_dis, (- node_dis, root))
            dis = node.val[root.split] - root.val[root.split]
            # 比较和分割平面的距离 判断是否在另外一颗子树上查找,同时也能决定是否继续查找节点
            if abs(dis) < - top_k_dis[0][0]:
                if dis < 0:
                    visit(root.right)
                else:
                    visit(root.left)

        visit(self.root)
        return top_k_dis

    def _distance(self, x, y):
        return np.linalg.norm(x - y, ord=self.p)

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
            print(cur.split('|'))
            val, label, split = cur.split('|')
            root.val = np.array(list((map(float, val[1: -1].split()))))
            root.label = int(label)
            root.split = int(split)
            root.left = self._deserialize(data)
            root.right = self._deserialize(data)
            return root

    def dump(self, model_path):
        """
        序列化保存二叉树
        :param model_path: 保存路径
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        tree_serialize = self._serialize(self.root, '')
        model = {'binary_tree': tree_serialize, 'k': self.k, 'p': self.p}
        pickle.dump(model, open(model_path + '/model', mode='wb'))

    def load(self, model_path):
        """
        加载二叉树
        :param model_path: 模型地址
        """
        model = pickle.load(open(model_path + '/model', mode='rb'))
        self.k = model['k']
        self.p = model['p']
        tree_serialize = model['binary_tree']
        tree_list = tree_serialize.split(',')
        self.root = self._deserialize(tree_list)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    model = KNN(5, 2)
    data = load_iris(return_X_y=True)
    size = data[0].shape[0]
    split = int(size * 0.8)
    X_train, y_train = data[0][:split], data[1][:split]
    X_test, y_test = data[0][split:], data[1][split:]
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    print(accuracy_score(y_test, predictions))
