#!/usr/bin/env python

# -*- encoding: utf-8

'''
                      ______   ___  __
                     / ___\ \ / / |/ /
                    | |    \ V /| ' / 
                    | |___  | | | . \ 
                     \____| |_| |_|\_\
 ==========================================================================
@author: Yekun Chai

@license: School of Informatics, Edinburgh

@contact: chaiyekun@gmail.com

@file: Ch2 KNN.py

@time: 09/10/2018 16:41 

@desc：       
               
'''

"""
K Nearest Neighbor
不需要训练
"""

import numpy as np
import operator


def createDataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    distances = (diffMat ** 2) ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCnt = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCnt[0][0]


if __name__ == '__main__':
    group, labels = createDataset()
    res = classify0([0, 0], group, labels, k=3)
    print(res)
