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

@file: Ch4_NBC.py

@time: 23/10/2018 20:22 

@desc：       
               
'''
"""
Naive Bayesian Classifier
# =======================
Pros: 数据较少情况仍然有效，可处理多类别问题 
Cons:对于输入数据的准备方式较为敏感

使用数据: 标称型数据

文档分类
"""


def loadDataSet():
    listPosts = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    listClasses = [0, 1, 0, 1, 0, 1]
    return listPosts, listClasses


# 构造vocabulary
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # set操作： | 并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


"""
问题：向量使用 1-hot存在问题，只考虑是否出现
改进：使用 Bag of Words 词袋模型
"""


# create sentence vector
# =====================
# def setOfWords2Vec(vocabList, inputSet):
#     # 判断是否出现，出现则置为1 （不考虑出现次数，BOW）
#     # 1-hot
#     sentVec = [0] * len(vocabList)
#     for word in inputSet:
#         if word in vocabList:
#             sentVec[vocabList.index(word)] = 1
#         else:
#             print("the word %s is out of vocab!" % word)
#     return sentVec
# =====================


def BagOfWords2Vec(vocabList, inputSet):
    # 判断是否出现，出现则置为1 （不考虑出现次数，BOW）
    # 1-hot
    sentVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # 统计出现次数
            sentVec[vocabList.index(word)] += 1
    return sentVec


# 训练
import numpy as np

"""
P(c_i|w) = P(w|c_i)P(c_i) / P(w)
"""


def trainNBC(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 计算P(c)
    prob_c = sum(trainCategory) / float(numTrainDocs)

    """
    初始化为0的问题：P(w|c=1) = P(w0|c=1)P(w1|c=1)P(w2|c=1)
    若某个值概率为0 （不存在），则乘积为0 。故改进： 初始化分母为2，所有词初始化次数为1
    """
    # ===================
    ### 初始化为0
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # p0Denom = p1Denom = .0
    # ===================
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = p1Denom = 2.0

    # 计算 p(word|c)
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    """
    问题： 下溢出，多个非常小的数字乘积
    解决办法：取自然对数 log
    """
    # ===================
    ### 下溢出，多个非常小的数字乘积
    # p1Vec = p1Num / p1Denom
    # p0Vec = p0Num / p0Denom
    # ===================
    p1Vec = np.log(p1Num / p1Denom)
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, prob_c


# 预测分类
def classifyNBC(vec2Classify, p0Vec, p1Vec, pClass1):
    # Bayesian公式两边同时取log，\Prod 变为 \Sigma
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    listPosts, listClasses = loadDataSet()
    # 建立 vocab
    myVocab = createVocabList(listPosts)
    # print(setOfWords2Vec(myVocab, listPosts[0]))

    # 词向量 (统计是否出现 1-hot)
    trainMat = []
    for postInDoc in listPosts:
        trainMat.append(BagOfWords2Vec(myVocab, postInDoc))

    # 训练NCBC
    p0V, p1V, P_c = trainNBC(trainMatrix=trainMat, trainCategory=listClasses)
    # print(p0V, p1V, P_c)

    # 预测分类
    testDocs = [['love', 'my', 'dalmation'], ['stupid', 'garbage']]
    for testDoc in testDocs:
        thisDocVec = np.array(BagOfWords2Vec(myVocab, testDoc))
        y_predict1 = classifyNBC(thisDocVec, p0V, p1V, P_c)
        print("{} is classified as: {}".format(testDoc, y_predict1))
