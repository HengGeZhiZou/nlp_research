# -*- encoding:utf-8 -*-
# @Time    : 2020/4/5 2:31 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from models.model import Model
import numpy as np


class Em(Model):
    """实现em优化算法，用来估计 GMM 中的参数"""

    def __init__(self, max_iter=200, mu_1=None, sigma_1=None, alpha_1=None, mu_2=None, sigma_2=None, alpha_2):
        """指定初始值"""
        super().__init__()
        self.max_iter = max_iter
        self.mu_1 = mu_1
        self.sigma_1 = sigma_1
        self.mu_2 = mu_2
        self.sigma_2 = sigma_2

    def train(self, train_data):
        for iter in range(self.max_iter):

    # 计算当前似然函数的值，判断是否达到收敛条件

    def predict(self, **args):
        pass

    def dump(self, **args):
        pass

    def load(self, **args):
        pass


if __name__ == '__main__':
    # 模拟混合高斯分布，设置为男性和女性身高
    # 男性：（mu=170，sigma=5.7）女性：（mu=160, sigma=5.2）,采样200人
    male_mu, male_sigma, male_alpha = 170, 5.7, 0.6
    female_mu, female_sigma, female_alpha = 160, 5.2, 0.4
    sample = 200
    male = np.random.normal(male_mu, male_sigma, int(sample * male_alpha))
    female = np.random.normal(female_mu, female_sigma, int(sample * female_alpha))
    train_data = np.append(male, female)
    np.random.shuffle(train_data)
    model = Em()
    model.train(train_data)
