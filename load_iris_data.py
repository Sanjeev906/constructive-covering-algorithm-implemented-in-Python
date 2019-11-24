#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import os.path
np.set_printoptions(suppress=True)


# In[2]:


ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
dataset_path = os.path.join(ROOT_DIR,r'data_set\iris.data')
dataset_out_path = os.path.join(ROOT_DIR,r'data_set\data_transformed\raw_dataset.out')


# In[3]:


label_set = (
    b'Iris-setosa',
    b'Iris-versicolor',
    b'Iris-virginica',
)
def read_label2(label):
    return label_set.index(label)


# In[4]:


raw_dataset = np.loadtxt(dataset_path, delimiter = ',', converters ={4:read_label2})
print(raw_dataset)


# In[5]:


np.savetxt(dataset_out_path, raw_dataset, delimiter=',')

