#!/usr/bin/python
# -*- encoding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 机器学习算法：线性回归--最小二乘法
def least_squares(data):
    # 1. 导入数据(data.csv), 返回一个二维数组
    points = np.genfromtxt(data, delimiter=',')
    # 提取points中的两列数据，分别作为x, y
    # print(points)
    # 取出所有的x
    xn = points[:, 0]
    # 取出所有的y
    yn = points[:, 1]
    # 绘制x, y的散点图
    plt.scatter(xn, yn)
    plt.show()
    # 通过拟合函数计算损失函数系数
    w, b = fit(points)
    print("w = ", w)
    print("b = ", b)
    print("cost = ", compute_cost(w, b, points))
    # 画出拟合曲线，plot是点图
    plt.scatter(xn, yn)
    # 针对每一个x计算出预测的y值
    pred_y = w * xn + b
    plt.plot(xn, pred_y, c='red')
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


def average(data):
    """
    求平均值
    :param data: 数据
    :return: 平均值
    """
    sum_data = 0
    m = len(data)
    for i in range(m):
        sum_data += data[i]

    return sum_data/m


def fit(points):
    """
    核心拟合函数，官方定义的方法名都是fit
    通过拟合函数计算出损失函数系数
    :param points: 数据
    :return: w, b
    """
    # 求和总次数
    m = len(points)
    # x的平均值
    x_avg = average(points[:, 0])
    # 计算w, b
    sum_yx = 0
    sum_xx = 0
    for i in range(m):
        x = points[i, 0]
        y = points[i, 1]
        # 对 y * (x - x_avg)求和
        sum_yx += y * (x - x_avg)
        # 对 x的平方求和
        sum_xx += x ** 2

    # 计算w的值
    w = sum_yx / (sum_xx - m * (x_avg ** 2))
    # 计算b的值
    sum_ywx = 0
    for i in range(m):
        x = points[i, 0]
        y = points[i, 1]
        sum_ywx += y - w * x

    b = sum_ywx / m
    return w, b


if __name__ == '__main__':
    least_squares('D:/Learn/Workspace/Python/machine-learning/src/data/data.csv')
