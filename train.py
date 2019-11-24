#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os.path
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
np.set_printoptions(suppress=True)
import json


# In[2]:


ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.join(ROOT_DIR,r'mod'))
from core import DataTrain
from datatest import DataTest


# In[3]:


raw_dataset_path = os.path.join(ROOT_DIR,r'data_set\data_transformed\raw_dataset.out')
full_result_path = os.path.join(ROOT_DIR,r'result\full\full_result.json')
train_minmax_path = os.path.join(ROOT_DIR,r'minmax_out\train_minmax.out')
test_minmax_path = os.path.join(ROOT_DIR,r'minmax_out\test_minmax.out')


# In[4]:


raw_dataset = np.loadtxt(raw_dataset_path, delimiter = ',')
# print(raw_dataset)


# In[5]:


m,n = raw_dataset.shape
print(m,n)


# In[6]:


# KFold划分
n_splits = 3


# In[7]:


# 数据类别数
classes = 7
full_result = {}
for i in range(classes):
    full_result[str(i)] = []
with open(full_result_path, 'w') as f_full_result:
    json.dump(full_result,f_full_result)


# In[8]:


lter = 1
kf = KFold(n_splits,shuffle=True)
for train_index,test_index in kf.split(raw_dataset):
    train_data = np.zeros([1,n])
    test_data = np.zeros([1,n])
    for i in train_index:
        row_temp = np.empty([1,n])
        for j in range(n):
            row_temp[0][j] = raw_dataset[i][j]
        train_data = np.append(train_data, row_temp, axis = 0)
    train_data = np.delete(train_data,0,axis = 0)
    # print(train_data)
    train_data_nontag = train_data[...,0:n-1]
    train_tag = train_data[...,n-1]
    tm = train_tag.size
    train_tag = train_tag.reshape(tm,1)
    # print(train_tag)
    min_max_scaler = preprocessing.MinMaxScaler().fit(raw_dataset[...,0:n-1])
    train_minmax = min_max_scaler.transform(train_data_nontag)
    train_minmax = np.concatenate((train_minmax,train_tag), axis=1)
    # print(train_minmax)
    np.savetxt(train_minmax_path, train_minmax, delimiter=',')
    train_process = DataTrain(lter,classes,train_minmax_path)
    train_process.start_train()
    # ------------------------------------------------------------------
    for i in test_index:
        row_temp = np.empty([1,n])
        for j in range(n):
            row_temp[0][j] = raw_dataset[i][j]
        test_data = np.append(test_data, row_temp, axis = 0)
    test_data = np.delete(test_data,0,axis = 0)
    # print(test_data)
    test_data_nontag = test_data[...,0:n-1]
    test_tag = test_data[...,n-1]
    tem = test_tag.size
    test_tag = test_tag.reshape(tem,1)
    # print(test_tag)
    test_minmax = min_max_scaler.transform(test_data_nontag)
    test_minmax = np.concatenate((test_minmax,test_tag), axis=1)
    # print(test_minmax)
    print('第' + str(lter) + '次迭代测试:')
    np.savetxt(test_minmax_path, test_minmax, delimiter=',')
    test_process = DataTest(classes,test_minmax_path,full_result_path)
    test_process.start_test()
    # ------------------------------------------------------------------
    # 合并结果
    present_result_path = ROOT_DIR + r'\result\result' + str(lter) + '.json'
    with open(present_result_path, 'r') as f_present_result:
        present_result = json.load(f_present_result)
    with open(full_result_path, 'r') as f_full_result:
        full_result = json.load(f_full_result)
    for cla in range(classes):
        full_result[str(int(cla))].extend(present_result[str(int(cla))])
    with open(full_result_path, 'w') as f_full_result:
        json.dump(full_result,f_full_result)
    # ------------------------------------------------------------------
    del train_process
    lter = lter + 1

