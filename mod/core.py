#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import os.path
import numpy as np
np.set_printoptions(suppress=True)
import math
import json

ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))

# In[1]:


class DataTrain:
    # 当前迭代数
    __lter = 0
    # 数据类别数
    __classes = 3
    # 训练数据目录
    __train_path = ''
    
    def __init__(self,l,c,p):
        self.__lter = l
        self.__classes = c
        self.__train_path = p

    def start_train(self):
        def add_simple(data,w,d,tag):
            count = 0
            m,n = data.shape
            for i in range(m):
                if np.dot(data[i][0:n-2],w) <= d and tag == data[i][n-2]:
                    data[i][n-1] = 1
                    count = count + 1
            return count

        def if_finish(train):
            m,n = train.shape
            for i in range(m):
                if train[i][n-1] == 0:
                    f = 0
                    break
            else:
                f = 1
            return f
        
        train = np.loadtxt(self.__train_path, delimiter = ',')

        # print(train)

        m,n = train.shape
        # print('原维度:%d %d' % (m,n))

        di =  []
        for i in range(m):
            temp = 0
            for j in range(n-1):
                temp = temp + math.pow(train[i][j],2)
            di.append(math.sqrt(temp))
        # print(di)

        dmax = math.ceil(max(di))
        for i in range(len(di)):
            di[i] = math.sqrt(math.pow(dmax,2) - math.pow(di[i],2))
        train = np.insert(train,4,di,axis=1)
        trans_m,trans_n = train.shape
        # print('升维后维度:%d %d' % (trans_m,trans_n))

        # print(train)

        # 最后一列记为是否已学习
        temp = np.zeros((trans_m,1))
        train = np.concatenate((train,temp),axis=1)
        # print(train)

        result = {}
        for i in range(self.__classes):
             result[str(i)] = []

        cc = 0
        while if_finish(train) == 0:
            if cc == 5000:
                print('///////////////////////////////超时！/////////////////////////////////')
                # print(train)
                break
            for i in range(trans_m):
                if train[i][trans_n] == 0:
                    train[i][trans_n] = 1
                    tag = train[i][trans_n-1]
                    temp = {}
                    temp['count'] = 1
                    w = train[i][0:trans_n-1]
                    temp['w'] = w.tolist()
                    d0 = []
                    d1 = []
                    for j in range(trans_m):
                        if train[j][trans_n-1] == tag:
                            d0.append(np.dot(w,train[j][0:trans_n-1]))
                        else:
                            d1.append(np.dot(w,train[j][0:trans_n-1]))
                    di1 = max(d1)
                    di0 = di1
                    d0.sort()
                    for z in range(len(d0)):
                        if d0[z] >= di1:
                            di0 = d0[z]
                            break
                    d = (di0 + di1)/2
                    temp['d'] = d
                    con = add_simple(train,w,d,tag)
                    temp['count'] = temp['count'] + con
                    result[str(int(tag))].append(temp)
            cc = cc + 1

        # print(train)

        result_path = ROOT_DIR + r'\result\result' + str(self.__lter) + r'.json'
        with open(result_path, 'w') as fw:
            json.dump(result,fw)
