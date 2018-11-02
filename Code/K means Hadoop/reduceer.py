#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
import os 


# In[ ]:


curr_cluster = None
number_of_points = 1
curr_gene = []
#centroid_sum = None
centroid_list = []


# f = open('mylog', 'w')

for line in sys.stdin:
    cluster_id, gene_id, features_string = line.strip().split('\t')
    features = np.fromstring(features_string, dtype='float', sep=',')
    
    if curr_cluster is None:
        curr_cluster = cluster_id
        curr_gene.append(gene_id)
        centroid_sum = [0]*len(features)
        centroid_sum += features
        
    elif curr_cluster == cluster_id:
        number_of_points +=1
        curr_gene.append(gene_id)
        centroid_sum = centroid_sum + features
    else:
        new_centroid = centroid_sum / float(number_of_points)
        #cent = cent + str(float("{0:.4f}".format(i))) + "\t"
        #new_centroid=np.array("{:0.4f}".format(x) for x in new_centroid) 
        new_centroid=np.round(new_centroid,4)
        centroid_list.append(new_centroid)
        print('%s\t%s\t' % (curr_cluster, curr_gene))
        curr_gene = []
        curr_cluster = cluster_id
        curr_gene.append(gene_id)
        number_of_points = 1
        centroid_sum = [0]*len(features)
        centroid_sum += features

	# f.write(line)
	# f.write("\n")
if curr_cluster == cluster_id:
    new_centroid = centroid_sum / float(number_of_points)
    new_centroid=np.round(new_centroid,4)
    centroid_list.append(new_centroid)
    print('%s\t%s\t' % (curr_cluster, curr_gene))
    curr_gene = []
    # f.write(centroid_list)
    # f.write("\n")

centroid_list = np.array(centroid_list)
centroid_df = pd.DataFrame(centroid_list)

centroid_df.to_csv('new_centroid',header=False,index=False)



    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    

