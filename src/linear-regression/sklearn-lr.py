#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 使用机器学习中的库实现线性回归
def sklearn_lr(data):
    # 1. 导入数据(data.csv), 返回一个二维数组
    points = np.genfromtxt(data, delimiter=',')
    # 2. 提取points中的两列，x, y
    x = points[:, 0]
    y = points[:, 1]
    # 3. 绘制散点图
    plt.scatter(x, y)
    plt.show()
    # 4. 创建LR对象
    lr = LinearRegression()
    x_new = x.reshape(-1, 1)
    y_new = y.reshape(-1, 1)
    lr.fit(x_new, y_new)
    # 从训练好的模型中提取系数以及截距
    print("w = ", lr.coef_)
    print("b = ", lr.intercept_)
    print("cost = ", compute_cost(lr.coef_, lr.intercept_, points))
    # 预测结果
    pred_y = lr.predict(x_new)
    plt.scatter(x, y)
    plt.plot(x, pred_y)
    plt.show()


def compute_cost(w, b, points):
    """
    损失函数：损失函数是系数的函数, 公式：(y - wx -b) ** 2 再对每个点的损失求和
    :param w: x的系数
    :param b: 常量
    :param points: 数据源
    :return:
    """
    # 每个点的损失和
    total_cost = 0
    # 点的总数
    m = len(points)
    # 逐点计算平方损失误差，然后求平均数
    for i in range(m):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2

    return total_cost/m


if __name__ == '__main__':
    sklearn_lr('D:/Learn/Workspace/Python/machine-learning/src/data/data.csv')
