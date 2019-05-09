# -*- coding: UTF-8 -*-
import numpy as np
import operator
import collections


def createDataSet():
    """
    函数说明:创建数据集
    Parameters:
        无
    Returns:
        group - 数据集
        labels - 分类标签
    """
    # 四组二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    #四组特征的标签
    labels = ['爱情片','爱情片','动作片','动作片']
    return group, labels



