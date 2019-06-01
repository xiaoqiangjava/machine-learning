#!/usr/bin/python
# -*- encoding:utf8 -*-
import numpy as np
import pandas as pds
import math


def tf_idf():
    """
    TF-IDF(Term Frequency-Inverse Document Frequency) 词频-逆文档频率：用于资讯检索与文档挖掘的常用加权计数
    :return:
    """
    # 定义数据和预处理
    doc_a = "The cat sat on my bed"
    doc_b = "The dog sat on my knees"
    bow_a = doc_a.split(' ')
    bow_b = doc_b.split(' ')
    # 构建词库
    word_set = set(bow_a).union(bow_b)
    print(word_set)
    # 用统计字典来保存每个词出现的次数
    word_dict_a = dict.fromkeys(word_set, 0)
    word_dict_b = dict.fromkeys(word_set, 0)
    # 遍历文档，统计次数
    for word in bow_a:
        word_dict_a[word] += 1

    for word in bow_b:
        word_dict_b[word] += 1

    df = pds.DataFrame([word_dict_a, word_dict_b])
    print(df)
    # 计算词频TF
    tf_a = cal_tf(word_dict_a, bow_a)
    tf_b = cal_tf(word_dict_b, bow_b)
    print(tf_a)
    print(tf_b)
    # 计算IDF
    idf_dict = cal_idf([word_dict_a, word_dict_b])
    print(idf_dict)
    # 计算两篇文档的TF-IDF
    tf_idf_a = cal_tf_idf(tf_a, idf_dict)
    tf_idf_b = cal_tf_idf(tf_b, idf_dict)
    df = pds.DataFrame([tf_idf_a, tf_idf_b])
    print(df)


def cal_tf(word_dict, bow):
    """
    计算词频: 词在文档中出现的次数 / 文档中单词总数
    :param word_dict: 单词在词库中出现的次数
    :param bow: 词带
    :return:
    """
    # 用一个对象记录tf, 把所有的词对应在bow文档里的tf都算出来
    tf_dict = {}
    # 计算tf：TF = per_count/total_count, 词语term在文档中出现的频率 = term在文档中出现的次数 / 文档中的单词总数
    term_count = len(bow)
    for term, count in word_dict.items():
        tf_dict[term] = count / term_count

    return tf_dict


def cal_idf(word_dicts):
    """
    计算每个词term的逆文档频率：idf = log((N + 1) / (M + 1)) N表示文档集中文档的总数，M表示文档集中包含了词term的文档数
    :return:
    """
    # 存储结果
    idf_dict = dict.fromkeys(word_dicts[0], 0)
    # 文档总数
    doc_num = len(word_dicts)
    # 求每个词的IDF
    for word_dict in word_dicts:
        # 求文档中包含词term的文档数
        for term, count in word_dict.items():
            if count > 0:
                idf_dict[term] += 1
    print('文档集中包含词的文档数：', idf_dict.__str__())

    for term, count in idf_dict.items():
        idf_dict[term] = math.log10((doc_num + 1) / (count + 1))

    return idf_dict


def cal_tf_idf(tf, idf):
    """
    计算TF-IDF值   TF-IDF = TF * IDF
    :param tf:
    :param idf:
    :return:
    """
    tf_idf = {}
    for term, tf_val in tf.items():
        tf_idf[term] = tf_val * idf[term]

    return tf_idf


if __name__ == '__main__':
    tf_idf()
