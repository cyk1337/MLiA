# !/usr/bin/env python

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

@file: Ch7_AdaBoost.py

@time: 24/10/2018 20:59

@desc：

'''
"""
集成方法Ensemble method， 元算法 （meta algorithm）

1. Bagging: (Boostrapping aggregating) 从原始数据集选择S次得到S个新数据集，分别训练S个分类器，预测时进行投票。(串联)
2. Boosting: 通过集中关注被已有分类器错分的数据来获得新分类器，分类结果给予所有分类器加权求和  （并联）

AdaBoost: Adaptive Boosting (自适应boosting)
Pros: 泛化错误率低，易编码
Cons: 对离群点敏感

"""

import numpy as np


def loadSimpData():
    dataMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 通过阈值比较对数据进行分类
def stumpClassifiy(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 单层决策树 （一个节点，一次分裂过程， decision stump, 树桩）
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(classLabels).T
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassifiy(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print("Split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f"
                      %
                      (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIter=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIter):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D", D.T)
        alpha = float(.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEstL", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("Total error:", errorRate)
        if errorRate == .0:
            break
    return weakClassArr


def adaClassify(dataToClass, classifierArr):
    datamatrix = np.mat(dataToClass)
    m = np.shape(datamatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassifiy(datamatrix, classifierArr[i]['dim'],
                                  classifierArr[i]['thresh'],
                                  classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    D = np.mat(np.ones((5, 1)) / 5)
    # res = buildStump(dataMat, classLabels, D)
    classifierArr = adaBoostTrainDS(dataArr=dataMat, classLabels=classLabels, numIter=9)
    testX = [[0, 0], [5, 5]]
    ys = adaClassify(testX, classifierArr)
    print(ys)
