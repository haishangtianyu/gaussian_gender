# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import math
from sklearn.metrics import confusion_matrix
import time

#加载数据集
def data_load(file_name):
    data_set = pd.read_csv(file_name)
    # 切片操作，第一个参数表示取全部的行，
    # 第二个参数x、y分别表示取第一个到最后一个，只取最后一个
    # x，y分别代表数据和标签
    x = data_set.values[:, :-1]
    y = data_set.values[:, -1]

    # 随机划分训练集与测试集，比例7:3
    train_num = random.sample(range(0, 3167), 2218) 
    # 设置随机数生成从0-3167中随机挑选3168*0.7=2218个随机数
    test_num = list(set(range(0, 3167)).difference(set(train_num)))

    # 测试集和训练集分道扬镳，7:3，互斥
    train_mat = np.array(x)[train_num]
    train_label = np.array(y)[train_num]

    test_mat = np.array(x)[test_num]
    test_label = np.array(y)[test_num]

    return train_mat, train_label, test_mat, test_label

#求高斯分布需要的参数并构建字典
def get_para(train_mat, train_label):
    #男声的序号
    male_list = []  
    #女声的序号
    female_list = []   
    male=female=0
    # 找到男声和女声的序号
    L_train = len(train_label)
    for i in range(L_train):
        if train_label[i] == 'male':
            male_list.append(i)
            male+=1
        else:
            female_list.append(i)
            female+=1

    continuousPara = {}
    for i in range(20):
        # 取出男声数据
        fea_data = train_mat[male_list, i]
        mean = fea_data.mean() #取第i个属性的均值
        std = fea_data.std() #计算第i个属性的标准差Standard deviation
        continuousPara[(i, 'male')] = (mean, std)
        # 取出女生数据
        fea_data = train_mat[female_list, i]
        mean = fea_data.mean()
        std = fea_data.std()
        continuousPara[(i, 'female')] = (mean, std)
    return continuousPara,male,female,L_train

#正态分布函数
def gaussian(x, mean, std):
    # 一维正态分布函数
    return 1 / (math.sqrt(math.pi * 2) * std) * math.exp((-(x - mean) ** 2) / (2 * std * std))

# 计算后验概率P(feature = x|C)，即各特征在已知性别的情况下的概率，x是该特征的值，feature_Index是特征在数组中的下标，c是分类男或女
def P(feature_Index, x, C, continuousPara):
    fea_para = continuousPara[(feature_Index, C)]
    mean = fea_para[0]
    std = fea_para[1]
    ans = gaussian(x, mean, std)
    return ans

#高斯贝叶斯过程
def Bayes(X, train_label, continuousPara,male,female,L_train):
    # 求先验概率
    male_para = male / L_train
    female_para = female / L_train
    # 朴素贝叶斯
    Result = []
    L_X = len(X)
    for i in range(L_X):
        ans_male = math.log(male_para)
        ans_female = math.log(female_para)
        L_Xi = len(X[i])
        for j in range(L_Xi):
            ans_male += math.log(P(j, X[i][j], 'male', continuousPara))
            ans_female += math.log(P(j, X[i][j], 'female', continuousPara))
        if ans_male > ans_female:
            Result.append('male')
        else:
            Result.append('female')
    return Result

#主模块
def main_():
    train_mat, train_label, test_mat, test_label = data_load('voice.csv')  #加载数据集
    continuousPara,male,female,L_train = get_para(train_mat, train_label)   #求高斯分布需要的参数
    predict_label = Bayes(test_mat, train_label, continuousPara,male,female,L_train)   #通过贝叶斯进行预测
    confusionMatrix = confusion_matrix(test_label, predict_label, labels = ['male', 'female']) #得出混淆矩阵,关于混淆矩阵参见https://blog.csdn.net/m0_38061927/article/details/77198990
    accu = []
    accu.append(confusionMatrix[0][0]/(1584-male))
    accu.append(confusionMatrix[1][1]/(1584-female))
    return accu

if __name__ == '__main__':
    accuracy_rate = []
    time_start = time.time()
    accuracy_rate = main_()
    time_end = time.time()
    print('男声正确率: %.2f      男声错误率: %.2f' %(accuracy_rate[0],1-accuracy_rate[0]))
    print('女声正确率: %.2f      女声错误率: %.2f' %(accuracy_rate[1],1-accuracy_rate[1]))
    print('时间: %fs' %(time_end - time_start))
