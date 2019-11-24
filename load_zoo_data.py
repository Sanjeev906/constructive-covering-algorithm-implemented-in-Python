#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import os.path
np.set_printoptions(suppress=True)


# In[2]:


ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
dataset_path = os.path.join(ROOT_DIR,r'data_set\zoo.data')
dataset_out_path = os.path.join(ROOT_DIR,r'data_set\data_transformed\raw_dataset.out')


# In[3]:


def read_label2(label):
    return str(int(label)-1)


# In[4]:


raw_dataset = np.loadtxt(dataset_path, delimiter = ',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17),converters ={17:read_label2})
np.delete(raw_dataset,0,axis = 1)
print(raw_dataset)


# In[5]:


np.savetxt(dataset_out_path, raw_dataset, delimiter=',')

