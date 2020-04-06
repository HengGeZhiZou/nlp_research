# -*- encoding:utf-8 -*-
# @Time    : 2020/3/23 4:40 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    """
    每个模型都需要继承当前类
    """

    def __init__(self):
        pass

    @abstractmethod
    def load(self, **args):
        """加载模型，通常为地址"""
        pass

    @abstractmethod
    def dump(self, **args):
        """保存模型到指定位置"""
        pass

    @abstractmethod
    def train(self, **args):
        pass

    @abstractmethod
    def predict(self, **args):
        pass
