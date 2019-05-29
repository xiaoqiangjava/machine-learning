#!/usr/bin/python
# -*- encoding=utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# 从sklearn中直接生成聚类数据
from sklearn.datasets.samples_generator import make_blobs
# 引入scipy中计算距离的库，默认欧式距离
from scipy.spatial.distance import cdist


class Kmeans(object):
    """
    K均值聚类算法
    需要的超参数：参数n_clusters(K), 迭代次数iter_num， 初始质心centroids
    """
    def __init__(self, n_clusters=6, iter_num=300, centroids=[]):
        self.n_clusters = n_clusters
        self.iter_num = iter_num
        self.centroids = np.array(centroids, dtype=np.float)

    def fit(self, data):
        """
        训练模型方法，K-means聚类过程
        :param data: data
        :return:
        """
        # 假设没有指定初始质心，就随机选取data中的点作为初始质心
        if self.centroids.shape == (0,):
            # 从data中随机选取0--data总行数的随机分类数目
            self.centroids = data[np.random.randint(0, data.shape[0], self.n_clusters), :]
        # 迭代分类
        for i in range(self.iter_num):
            # 1. 计算距离,100 * 6
            distances = cdist(data, self.centroids)
            # 2. 使用K-nn思想聚类，对距离由近到远排序，选取最近的质心点的类别，作为当前点的分类, 100 * 1
            cluster_index = np.argmin(distances, axis=1)
            # 3. 对每一类数据进行均值计算，更新质心点
            for cluster in range(self.n_clusters):
                # 排除掉没有在cent_index的类别
                if cluster in cluster_index:
                    # 选出所有类别是cluster的点，取data里面坐标的均值，更新cluster质心点, 选取时使用了布尔索引
                    # axis指定y轴方向，得到的是一个点, 对每一列求均值
                    self.centroids[cluster] = np.mean(data[cluster == cluster_index], axis=0)

    def predict(self, samples):
        """
        预测， 先计算距离，然后选取距离最近的那个质心的类别
        :param samples:
        :return:
        """
        distances = cdist(samples, self.centroids)
        cluster_index = np.argmin(distances, axis=1)
        return cluster_index


def k_means():
    """
    无监督学习--k均值算法
    :return:
    """
    # 从sklearn中直接生成数据
    x, y = make_blobs(n_samples=100, centers=6, random_state=1234, cluster_std=0.6)
    plt.figure(figsize=(6, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()
    # 测试算法
    kmeans = Kmeans(centroids=np.array([[2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6]]))
    # 绘制初始状态时的散点图
    plt.figure(figsize=(16, 6))
    plot_k_means(x, y, kmeans.centroids, 121, 'Init state')
    # 开始聚类
    kmeans.fit(x)
    plot_k_means(x, y, kmeans.centroids, 122, 'Final state')
    # 预测新数据点的类别
    x_test = np.array([[10, 7], [0, 0]])
    y_pred = kmeans.predict(x_test)
    print(y_pred)
    print(kmeans.centroids)
    plt.scatter(x_test[:, 0], x_test[:, 1], s=100, c='black')
    plt.show()


def plot_k_means(x, y, centroids, subplot, title):
    """
    定义一个绘制子图工具
    :param x: x
    :param y: y
    :param centroids: 质心点
    :param subplot: 子图
    :param title: title
    :return:
    """
    # 绘制样本点
    # 分配子图，121表示1行2列中子图的第一个
    plt.subplot(subplot)
    plt.scatter(x[:, 0], x[:, 1], c='r')
    # 画出质心点
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.array(range(6)), s=100)
    plt.title(title)


if __name__ == '__main__':
    k_means()
