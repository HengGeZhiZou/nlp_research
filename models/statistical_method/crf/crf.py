# -*- encoding:utf-8 -*-
# @Time    : 2020/4/13 6:14 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from models.model import Model


class CRF(Model):
    """条件随机场一般接在全连接层之后，这里假设输入为上一层网络的输出"""

    def __init__(self):
        super().__init__()

    def train(self, **args):
        pass

    def predict(self, **args):
        pass

    def dump(self, **args):
        pass

    def load(self, **args):
        pass


if __name__ == '__main__':
    pass
