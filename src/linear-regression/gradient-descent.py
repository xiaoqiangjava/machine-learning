#!/usr/bin/python
# -*- coding:utf-8 -*_
import numpy as np
import matplotlib.pyplot as plt


# 机器学习算法：线性回归--梯度下降法
def gradient_descent(data):
    # 1. 导入数据(data.csv), 返回一个二维数组
    points = np.genfromtxt(data, delimiter=',')
    # 2. 提取points中的两列，x, y
    x = points[:, 0]
    y = points[:, 1]
    # 3. 绘制散点图
    plt.scatter(x, y)
    plt.show()
    # 4. 定义模型超参数
    num_iter = 10
    alpha = 0.0001
    init_w = 0
    init_b = 0
    # 5. 调用梯度下降函数计算w, b
    w, b, cost_list = fit(points, num_iter, init_w, init_b, alpha)
    print("w = ", w)
    print("b = ", b)
    print("cost = ", compute_cost(w, b, points))
    # 绘制逐步迭代的损失函数图像
    plt.plot(cost_list)
    plt.show()
    # 测试
    pred_y = w * x + b
    plt.scatter(x, y)
    plt.plot(x, pred_y, c='red')
    plt.show()


def compute_cost(w, b, points):
    """
    损失函数：跟最小二乘法相同，都是计算平方损失误差
    :param w: x的系数，不同的是此时一个x对应一个系数
    :param b: 常量
    :param points: 数据点
    :return: total_cost
    """
    total_cost = 0
    num = len(points)
    for i in range(num):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2

    return total_cost / num


def fit(points, num_iter, w, b, alpha):
    """
    拟合函数，使用梯度下降法，需要提前指定起始点，步长，迭代次数
    :param points: 数据
    :param num_iter: 迭代次数
    :param w: w初始值
    :param b: b初始值
    :param alpha: 步长
    :return: (w, b, cost_list)
    """
    # 定义一个list，保存所有的损失函数值，用来显示下降的过程
    cost_list = []
    # 迭代计算w, b
    for i in range(num_iter):
        cost_list.append(compute_cost(w, b, points))
        w, b = step_grad_desc(w, b, points, alpha)

    return w, b, cost_list


def step_grad_desc(curr_w, curr_b, points, alpha):
    """
    单步梯度下降
    :param curr_w: 当前w
    :param curr_b: 当前b
    :param points: 数据
    :param alpha: 步长
    :return:
    """
    total_grad_w = 0
    total_grad_b = 0
    num = len(points)
    # 带入公式
    for i in range(num):
        x = points[i, 0]
        y = points[i, 1]
        total_grad_b += curr_w * x + curr_b - y
        total_grad_w += (curr_w * x + curr_b - y) * x

    # 梯度下降之后的w, b
    update_w = curr_w - alpha * 2 * total_grad_w / num
    update_b = curr_b - alpha * 2 * total_grad_b / num

    return update_w, update_b


if __name__ == '__main__':
    gradient_descent('D:/Learn/Workspace/Python/machine-learning/src/data/data.csv')
