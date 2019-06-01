#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


def lfm():
    """
    LFM隐语义模型梯度下降法求解
    输入参数：
    R: m * n的评分矩阵
    k: 隐特征向量维度
    max_iter： 最大迭代次数
    alpha： 步长
    lamda： 正则化系数
    输出参数：
    分解之后的P, Q
    P: 初始化用户特征矩阵 m * k
    Q: 初始化物品特征矩阵 n * k
    """
    # 定义评分矩阵, 6个User，5个Item
    R = np.array([[4, 0, 2, 0, 1],
                  [0, 2, 3, 0, 0],
                  [1, 0, 2, 4, 0],
                  [5, 0, 0, 3, 1],
                  [0, 0, 1, 5, 1],
                  [0, 3, 2, 4, 1]])

    print(R.shape)
    # 定义模型超参数
    k, max_iter, alpha, lamda = 5, 5000, 0.0002, 0.004
    # 测试
    P, Q, cost = lfm_grad_desc(R, k, max_iter, alpha, lamda)
    # 得到预测矩阵
    pred_R = np.dot(P, Q.T)
    print(P)
    print(Q)
    print(cost)
    print(R)
    print(pred_R)


def lfm_grad_desc(R, k, max_iter, alpha=0.0001, lamda=0.004):
    """
    使用梯度下降法求解LFM隐语义模型
    :return:
    """
    # 基本维度参数定义
    m = R.shape[0]
    n = R.shape[1]

    # 分解矩阵P, Q初始值随机生成, P是m * k矩阵，Q是k * n矩阵
    P = np.random.rand(m, k)
    Q = np.random.rand(n, k)
    # 将Q做转置，用于求矩阵的乘积
    Q = Q.T
    # 开始迭代
    for step in range(max_iter):
        # 对所有的用户u和物品i遍历，对应的特征向量Pu, Qi梯度下降
        for u in range(m):
            for i in range(n):
                # 对每个大于0的评分，求出评分预测误差
                if R[u][i] > 0:
                    # 用户u对物品i的评分误差, np.dot是向量的点乘
                    e_ui = np.dot(P[u, :], Q[:, i]) - R[u][i]
                    # 带入梯度下降公式，迭代计算Pu, Qi
                    for k_index in range(k):
                        # 计算得到P中的每个元素, 对所有的误差求和相当于减去每次计算得到的误差
                        P[u][k_index] = P[u][k_index] - alpha * (2 * e_ui * Q[k_index][i] + 2 * lamda * P[u][k_index])
                        Q[k_index][i] = Q[k_index][i] - alpha * (2 * e_ui * P[u][k_index] + 2 * lamda * Q[k_index][i])
        # u, i遍历完成，所有的特征向量更新完成，得到P, Q，可以计算预测评分矩阵
        pred_R = np.dot(P, Q)
        # 计算当前损失函数
        cost = 0
        for u in range(m):
            for i in range(n):
                if R[u][i] > 0:
                    cost += (np.dot(P[u, :], Q[:, i]) - R[u][i]) ** 2
                    # 加上正则化系数想
                    for k_index in range(k):
                        cost += lamda * (P[u, k_index] ** 2 + Q[k_index, i] ** 2)

        if cost < 0.0001:
            break
    return P, Q.T, cost


if __name__ == '__main__':
    lfm()

