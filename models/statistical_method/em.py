# -*- encoding:utf-8 -*-
# @Time    : 2020/4/5 2:31 下午
# @Author  : HengGeZhiZou <1018676477@qq.com>
from models.model import Model
import numpy as np


class Em(Model):
    """实现em优化算法，用来估计 GMM 中的参数"""

    def __init__(self, max_iter=1000, mu_1=None, sigma_1=None, alpha_1=None, mu_2=None, sigma_2=None, alpha_2=None):
        """指定初始值"""
        super().__init__()
        self.max_iter = max_iter
        self.mu_1 = mu_1
        self.sigma_1 = sigma_1
        self.alpha_1 = alpha_1
        self.mu_2 = mu_2
        self.sigma_2 = sigma_2
        self.alpha_2 = alpha_2

    def _guass(self, y, mu, sigma):
        """
        计算高斯分布的结果
        :param y: 数据的分布
        :param mu: 均值
        :param sigma 方差
        :return: 密度函数的结果
        """
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- ((y - mu) ** 2) / (2 * (sigma ** 2)))

    def _log_likelihood(self, y, gamma_1, gamma_2):
        """
        计算完全对数似然值的大小
        :param train_data: 训练数据
        :return: 完全对数似然
        """
        log_1 = sum(gamma_1) * np.log(self.alpha_1) + np.dot(gamma_1, (
                np.log(1 / np.sqrt(2 * np.pi)) - np.log(self.sigma_1) - (1 / (2 * self.sigma_1 ** 2)) * (
                y - self.mu_1) ** 2))
        log_2 = sum(gamma_2) * np.log(self.alpha_2) + np.dot(gamma_2, (
                np.log(1 / np.sqrt(2 * np.pi)) - np.log(self.sigma_2) - (1 / (2 * self.sigma_2 ** 2)) * (
                y - self.mu_2) ** 2))
        return log_1 + log_2

    def train(self, train_data):
        pre_log_likelihood = 0
        size = train_data.shape[0]
        gamma_1 = np.zeros(size)
        gamma_2 = np.ones(size)
        for _ in range(self.max_iter):
            # 判断是否达到停止的条件
            cur_log_likelihood = self._log_likelihood(train_data, gamma_1, gamma_2)
            print(cur_log_likelihood)
            if abs(cur_log_likelihood - pre_log_likelihood) < 1e-2:
                print('停止迭代')
                break
            # E步 计算每个数据在不同分布下的响应度
            gamma_1 = self.alpha_1 * self._guass(train_data, self.mu_1, self.sigma_1)
            gamma_2 = self.alpha_2 * self._guass(train_data, self.mu_2, self.sigma_2)
            nor = gamma_1 + gamma_2
            gamma_1 = gamma_1 / nor
            gamma_2 = gamma_2 / nor
            # M步，求最大化似然函数的参数
            self.mu_1 = np.dot(gamma_1, train_data) / sum(gamma_1)
            self.mu_2 = np.dot(gamma_2, train_data) / sum(gamma_2)
            self.sigma_1 = np.sqrt(np.dot(gamma_1, (train_data - self.mu_1) ** 2) / sum(gamma_1))
            self.sigma_2 = np.sqrt(np.dot(gamma_2, (train_data - self.mu_2) ** 2) / sum(gamma_2))
            self.alpha_1 = sum(gamma_1) / size
            self.alpha_2 = sum(gamma_2) / size
            pre_log_likelihood = cur_log_likelihood

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
    # 设定初始值
    model = Em(mu_1=168, sigma_1=5, alpha_1=0.5, mu_2=158, sigma_2=5, alpha_2=0.5)
    model.train(train_data)
    print(model.alpha_1, model.sigma_1, model.mu_1)
    print(model.alpha_2, model.sigma_2, model.mu_2)
