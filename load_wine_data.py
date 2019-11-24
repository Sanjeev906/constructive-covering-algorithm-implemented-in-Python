#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import os.path
np.set_printoptions(suppress=True)


# In[2]:


ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
dataset_path = os.path.join(ROOT_DIR,r'data_set\wine.data')
dataset_out_path = os.path.join(ROOT_DIR,r'data_set\data_transformed\raw_dataset.out')


# In[3]:


def read_label2(label):
    return str(int(label)-1)


# In[4]:


raw_dataset = np.loadtxt(dataset_path, delimiter = ',',converters ={0:read_label2})
raw_dataset_label = raw_dataset[...,0]
rm = raw_dataset_label.size
raw_dataset_label = raw_dataset_label.reshape(rm,1)
raw_dataset = np.concatenate((raw_dataset[...,1:],raw_dataset_label),axis = 1)
print(raw_dataset)


# In[5]:


np.savetxt(dataset_out_path, raw_dataset, delimiter=',')

