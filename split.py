#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split

#load dataset
df = read_csv('cs-training.csv', index_col=0)

#split into train and test
train2, test2 = train_test_split(df, test_size=0.50, random_state=42)

#save train&test sets

train2.to_csv('train2.csv')
test2.to_csv('test2.csv')

