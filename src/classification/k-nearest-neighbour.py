#!/usr/bin/python
# -*- encoding:utf-8 -*-
import numpy as np
import pandas as pds
# 直接引入sklearn里的数据集，iris鸢尾花
from sklearn.datasets import load_iris
# 切分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Knn(object):
    """
    定义一个类，封装k-nearest-neighbout算法：
    KNN是一个分类算法，它的思想是找到与测试数据距离最近的K个训练数据，求出前K个数据所在的类别出现的
    频率，将前K个点中出现频率最高的类别作为预测数据的类别，因此为了保证前K个点中有最多的分类， K的取
    值一般是奇数。
    """
    def __init__(self, k, distance):
        """
        初始化操作，KNN算法需要指定k, 以及求距离的算法
        :param k: 最近邻域数
        :param distance: 求解距离函数
        """
        self.k = k
        self.distance = distance

    def fit(self, x, y):
        """
        训练模型，KNN算法训练模型不需要处理，只要拿到训练数据即可，在预测时求测试数据与
        训练数据之间的距离
        :param x 训练数据中的x
        :param y 训练数据中的y
        :return:
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """
        分类，预测结果
        1> 计算测试数据与训练数据之间的距离
        2> 按照距离的递增关系进行排序
        3> 选取距离最小的k个点
        4> 求出前K个点各个类别出现的频率
        5> 返回频率最高的类别
        :param x: 测试数据
        :return: 预测结果集
        """
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)
        # 迭代测试数据集，计算每一行对应的类别
        for i, x_test in enumerate(x):
            # 计算x_test与每个训练数据中的距离
            distances = self.distance(self.x_train, x_test)
            # 对距离从近到远排序
            nn_index = np.argsort(distances)
            # 取出前K个最近邻对应的分类
            nn_y = self.y_train[nn_index[:self.k]].ravel()
            # 统计类别中出现频率最高的那个分类，赋值给y_pred[i]
            y_pred[i] = np.argmax(np.bincount(nn_y))

        return y_pred

    @classmethod
    def l1_distance(cls, x_train, x_test):
        """
        求曼哈顿距离
        对没一行测试数据，求出所有训练数据与该测试数据的距离，然后求和
        :param x_train: 训练数据集
        :param x_test: 训练数据集，是一行数据
        :return: distance
        """
        return np.sum(np.abs(x_train - x_test), axis=1)

    @classmethod
    def l2_distance(cls, x_train, x_test):
        """
        求欧式距离
        :param x_train: 训练数据
        :param x_test: 测试数据
        :return: distance
        """
        return np.sqrt(np.sum((x_train - x_test) ** 2, axis=1))


def knn():
    # 加载数据
    iris = load_iris()
    df = pds.DataFrame(data=iris.data, columns=iris.feature_names)
    df['class'] = iris.target
    df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})
    print(df)
    # 提取监督学习中的x, y, 其中x.shape=(150, 4), y.shape(150,)
    x = iris.data
    y = iris.target
    # 将数据分为测试数据和训练数据, stratify参数指定类别数组，将按照类别将数据均匀分成训练数据集合测试数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50, stratify=y)

    # 测试
    result = []
    for i in range(2):
        dis_func = Knn.l1_distance if i == 1 else Knn.l2_distance
        for k in range(1, 10, 2):
            knn = Knn(k, dis_func)
            # 训练模型
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            # 计算预测准确率
            accuracy = accuracy_score(y_test, y_pred)
            result.append((dis_func.__name__, k, accuracy))

    df = pds.DataFrame(result, columns=['dis_func', 'k', 'accuracy'])
    print(df)


if __name__ == '__main__':
    knn()
