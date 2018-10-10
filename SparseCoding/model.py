# -*- coding: utf-8 -*-

"""
--------------------------------------------
File Name:          model
Author:             deepgray
--------------------------------------------
Description:
稀疏编码的模型:

    1. 基本的ISTA算法
    2. 二维的情况
--------------------------------------------
Date:               18-6-29
Change Activity:

--------------------------------------------
"""
import numpy as np
import numpy.matlib
from numpy import linalg as la


def soft_thres(x, theta):
    """
    软阈值函数：soft_{theta}(x)
    :param x:
    :param theta:
    :return:
    """
    return np.sign(x) * np.fmax(0, np.abs(x)-theta)


def ista(A, x, lamb, alpha=None, k=None, maxiter=1500, eps=1e-6):
    """
    基础的ISTA算法，收敛条件为1/alpha 大于D^T*D的最大特征值
    :param A: 字典
    :param x: 输入向量
    :param lamb: 超参数lambda
    :param alpha: 参数，学习率（梯度下降的更新率）
    :param k: 迭代次数
    :param maxiter: 最大迭代次数
    :param eps: 迭代终止的误差最小值
    :return: 编码h
    """
    num_iter = k if k is not None else maxiter
    if alpha is None:
        alpha = 1/(np.max(la.eigh(np.matmul(A.T, A))[0]))

    h_0 = np.zeros(A.shape[1])
    h = [h_0]

    for k in range(num_iter):
        z = h[k] - alpha * np.matmul(A.T, np.matmul(A, h[k])-x)
        h.append(soft_thres(z, lamb*alpha))
        if la.norm(h[k+1] - h[k]) < eps:
            break
    # print(h[-1].shape)
    return h


# def gradient_soft_thres(x, t):
#     """
#     软阈值函数的梯度
#     :param x:
#     :param t:
#     :return:
#     """
#     return (x > t) + (x < t)


def sgd_dictionary(A, x, h, alpha, t):
    """
    使用梯度下降优化字典
    # TODO 这个地方可以改成K-SVD算法
    :param A: 字典
    :param x: 输入
    :param h: 编码表示
    :param alpha: 学习率
    :param t: 迭代次数
    :return: 更新后的字典
    """
    # print(A.shape, h.shape)
    temp = np.matmul(A, h) - x
    temp = np.reshape(temp, (temp.shape[0], 1))
    col = np.matlib.repmat(temp, 1, A.shape[1])
    #  print(col.shape)
    temp = np.matlib.repmat(np.reshape(h, (h.shape[0], 1)), 1, A.shape[1])
    dA = temp * col

    new = A - alpha * dA
    new -= np.mean(new)
    new /= np.std(new, axis=0)
    return new


class SparseCode(object):
    """
    稀疏编码模型
    """
    def __init__(self, dict_size, lamb, alpha, k=None):
        """
        稀疏编码模型
        :param dict_size: 字典大小
        :param lamb: lambda的取值
        :param alpha: 梯度更新率alpha的取值
        """
        self.dict_size = dict_size
        self.lamb = lamb
        self.alpha = alpha
        self.k = k
        A = np.random.randn(dict_size[0], dict_size[1])
        A -= np.mean(A)
        # normalization of A:
        self.A = A/np.std(A, axis=0)

    def train(self, input_vec, iteration=1000):
        """
        多步训练
        :param input_vec:
        :param iteration:
        :return:
        """
        assert len(input_vec) == iteration

        for i in range(iteration):
            h = ista(self.A, input_vec[i], self.lamb, self.alpha, self.k)[-1]
            # print(input_vec[i].shape, h.shape)
            new = sgd_dictionary(self.A, input_vec[i], h, 0.001, i)
            self.A = new

    def save(self, path):
        np.save(path, self.A)
  
    def load(self, path):
        self.A = np.load(path)

    def test(self, input_x, iteration=10000):
        """
        多步测试
        :param input_x:
        :param iteration:
        :return:
        """
        assert len(input_x) == iteration

        h = list()
        for i in range(iteration):
            h0 = ista(self.A, input_x[i], self.lamb, self.alpha, self.k)[-1]
            h.append(h0)
        return h

    def get_size(self):
        return self.dict_size

    def train_one_step(self, input_x):
        """
        一步训练
        :param input_x:
        :return:
        """
        h = ista(self.A, input_x, self.lamb, self.alpha, self.k)
        new = sgd_dictionary(self.A, input_x, h[-1], self.alpha, i)
        self.A = new

    def test_one_step(self, input_x):
        """
        一步训练
        :param input_x:
        :return:
        """
        return ista(self.A, input_x[i], self.lamb, self.alpha, self.k)[-1]

    def compute_anomaly(self, h, input_x):
        """
        计算重构误差
        :param h:
        :param input_x:
        :return:
        """
        error = list()
        size = len(h)
        assert size == len(input_x)

        for i in range(size):
            err = np.matmul(self.A, h[i]) - input_x[i]
            error.append(la.norm(err))
        return error

    def compute_anomaly_one_step(self, code, input_x):
        error = np.matmul(self.A, code) - input_x
        return la.norm(error)

    def get_dict(self):
        return self.A
