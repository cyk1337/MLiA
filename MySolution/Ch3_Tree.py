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

@file: Ch3_Tree.py.py

@time: 23/10/2018 18:10 

@desc：       
               
'''
"""
Decision Tree
"""
import math


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # 记录每一类数量
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = .0
    # 计算信息熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log2(prob)
    return shannonEnt


def createDataset():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']]
    labels = ['no sufacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    # 将选中作为分类的属性 axis 以及 value 分开，并且保存剩余特征
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureTopSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 决策树父节点 原始的 熵 H(A)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = .0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [sample[i] for sample in dataSet]
        uniqueVals = set(featList)
        newEntropy = .0
        # 计算 条件熵 H(A|B)
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


"""
递归构造决策树
"""

import operator


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # return the maximum count
    return sortedClassCount[0][0]


import copy


def createTree(dataSet, labels):
    # 由于是传地址引用，因此先深拷贝 （减少label后原labels不受影响）
    labels = copy.deepcopy(labels)
    classList = [sample[-1] for sample in dataSet]
    # 只有一类, 即类别完全相同， 则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时 返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureTopSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    # 得到列表包含的所有属性列表
    featValues = [sample[bestFeat] for sample in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # label字符串转为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    myDataset, labels = createDataset()
    # split branch of decision tree -------
    # ds = splitDataSet(myDataset, 0, 0)
    # calc entropy --------------
    # entropy = calcShannonEnt(dataSet=myDataset)
    # res = chooseBestFeatureTopSplit(myDataset)

    # 构造决策树
    myTree = createTree(myDataset, labels)
    print("Tree:", myTree)
    # 预测！
    testVecs = [[1, 0], [0, 0], [1, 1]]
    for testVec in testVecs:
        y = classify(myTree, labels, testVec=testVec)
        print(y)

    # ===================
    # 决策树可以使用pickle存储
    # ===================
