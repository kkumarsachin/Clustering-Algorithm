#!/usr/bin/env python3
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from scipy.spatial import distance
import os 
import sys


# In[26]:


centroids = pd.read_csv('centroid',header=None).values
x = pd.read_csv('new_dataset_1.txt', sep='\t',header=None)
gene = list(x[0].values)
ground_truth = list(x[1].values)
data = x.drop([0,1],axis=1).values


# In[27]:



for idx, point in enumerate(data):
    min_dist=float('inf')
    ps = ','.join(str(variable) for variable in point)  
    #cluster_id = np.argmin([distance.euclidean(cent,point) for cent in centroids])
    for i in range(centroids.shape[0]):
        if distance.euclidean(point,centroids[i]) <= min_dist:
            min_dist=distance.euclidean(point,centroids[i])
            cluster_id = i
    print('%s\t%s\t%s' % (cluster_id, gene[idx], ps))

