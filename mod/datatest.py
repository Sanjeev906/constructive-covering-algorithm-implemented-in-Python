#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.set_printoptions(suppress=True)
import math
import json


# In[2]:

class DataTest:
    # 数据类别数
    __classes = 3
    # 测试数据路径
    __test_minmax_path = 'E:\AI\test_minmax.out'
    # 训练结果路径
    __full_result_path = 'E:\AI\result\full\full_result.json'
    
    def __init__(self,c,tp,fp):
        self.__classes = c
        self.__test_minmax_path = tp
        self.__full_result_path = fp
        
    def start_test(self):
        
        def gravitation(C,present_test):
            m = len(C['w'])
            culmut = 0.0
            for i in range(m):
                culmut = culmut + math.pow((C['w'][i]-present_test[i]),2)
            result = C['count']/culmut
            return round(result,2)
            
        
        test_data = np.loadtxt(self.__test_minmax_path, delimiter = ',')
        with open(self.__full_result_path, 'r') as f_full_result:
            full_result = json.load(f_full_result)

        m,n = test_data.shape
        # print('原维度:%d %d' % (m,n))

        di =  []
        for i in range(m):
            temp = 0
            for j in range(n-1):
                temp = temp + math.pow(test_data[i][j],2)
            di.append(math.sqrt(temp))
           # print(di)

        dmax = math.ceil(max(di))
        for i in range(len(di)):
             di[i] = math.sqrt(math.pow(dmax,2) - math.pow(di[i],2))
        test_data = np.insert(test_data,4,di,axis=1)
        test_m,test_n = test_data.shape
        # print('升维后维度:%d %d' % (test_m,test_n))

        out = np.zeros([1,self.__classes])
        predict_result = []
        total_match = 0
        for testi in range(m):
            success = []
            outi = []
            present_test = test_data[testi,0:test_n-1]
            present_tag = int(test_data[testi,test_n-1])
            for i in range(self.__classes):
                oi = 0
                for j in full_result[str(int(i))]:
                    w = np.array(j['w'])
                    w = w.reshape(1,test_n-1)
                    if np.dot(w,present_test) >= j['d']:
                        oi = oi + 1
                outi.append(oi)
            for temp in outi:
                if temp != 0:
                    break;
            else:
                del outi
                outi = []
                flag = 0
                for i in range(self.__classes):
                    oi = 0
                    for j in full_result[str(int(i))]:
                        oi = oi + gravitation(j,present_test)
                    outi.append(oi)
            predict = outi.index(max(outi))
            out = np.append(out,np.asarray(outi).reshape(1,self.__classes),axis=0)
            predict_result.append(predict)
            if predict == present_tag:
                success.append(1)
                total_match = total_match + 1
            else:
                success.append(0)
        print('正确率: %.2f %%' % ((total_match/m)*100))

        # 对测试集所有数据的预测结果
        print('对测试集所有数据的测试结果')
        out = np.delete(out,0,axis=0)
        print(out)
        print('------------------------------------------------------------------')
        print('对测试集所有数据的预测结果')
        print(predict_result)
        print('==================================================================')
