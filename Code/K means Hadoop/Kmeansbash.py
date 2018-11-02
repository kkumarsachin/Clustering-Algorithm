#!/usr/bin/env python3	
# coding: utf-8

# In[61]:

import os 
import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance
import seaborn as sns
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.switch_backend('Agg')
# In[62]:


def jaccard_rand(predicted, ground):
    length = len(predicted) 
    predicted_matrix = np.zeros((length, length))
    ground_matrix = np.zeros((length, length))
    
    ## Ground truth and Predicted matrix
    for i in range(length):
        for j in range(length):
            if predicted[i]==predicted[j]:
                predicted_matrix[i][j]=1
            if ground[i]==ground[j]:
                ground_matrix[i][j] = 1
                
    ## Calculating the agree and disagree value for jaccard and rand
            
    m00,m01,m10,m11 = 0,0,0,0
    for i in range(length):
        for j in range(length):
            if predicted_matrix[i][j]+ground_matrix[i][j]==2:
                m11 +=1
            elif predicted_matrix[i][j]+ground_matrix[i][j]==0:
                m00 +=1
            elif predicted_matrix[i][j]==0 and ground_matrix[i][j]==1:
                m01 +=1
            elif predicted_matrix[i][j]==1 and ground_matrix[i][j]==0:
                m10 +=1
                
    ## Calculating jaccard and rand index
    jaccard = float(m11)/(m11 + m10 + m01)
    rand = float(m11 + m00) / (m11 + m10 + m01 + m00)

    return jaccard, rand


# In[63]:


def plot_pca(predicted_clusters,X):
    pca = PCA(n_components=2)
    pca.fit(X)
    x_pca = pca.transform(X)
    pca_x = pd.DataFrame(x_pca,columns=['pc_1','pc_2'])
    pca_x['clusters'] = predicted_clusters
    sns.set_style('darkgrid')
    sns.lmplot(x = 'pc_1',y='pc_2',data = pca_x,hue='clusters',fit_reg=False,height=7,aspect=1)
    plt.title('K Means Clustering')
    plt.savefig('some.png')
    plt.show()
    


# In[ ]:


x = pd.read_csv('new_dataset_1.txt', sep='\t', header=None)
gene = list(x[0].values)
ground_truth = list(x[1].values)
data = x.drop([0,1],axis=1)
Y = data.values
data = pd.DataFrame(data)
rows,columns = data.shape
k = 3
# print(x)

# In[65]:


np.random.seed(101)
# cents = sorted(np.random.randint(0,rows,size=k))
cents = [3, 5, 9]
centroid = [data.iloc[cent].values for cent in cents]
centroid = np.array(centroid)
centroid_df = pd.DataFrame(centroid)
centroid_df.to_csv('centroid',header=False,index=False)


# In[33]:


os.system("hdfs dfs -rm -r kminput")
os.system("hdfs dfs -mkdir kmout")
os.system("hdfs dfs -put new_dataset_1.txt kminput")


# In[ ]:
iteration = 0
f=open('minus_log','w')
while True:
    os.system("hdfs dfs -rm -r kmout")
    os.system("hdfs dfs ")
    os.system("hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.9.1.jar -mapper mapper.py -reducer reduceer.py -input kminput -output kmout")
    new_centroids = pd.read_csv('new_centroid',header=None)
    new_centroids = new_centroids.values
    new_centroids=np.round(new_centroids,4)
    centroid=pd.read_csv('centroid',header=None).values
    centroid=np.round(centroid,4)
    f.write(str(centroid - new_centroids))
    test=[]

    if np.sum(centroid-new_centroids)==0 or iteration == 10:
        break
    new_c=pd.DataFrame(new_centroids)
    new_c.to_csv('centroid',header=False,index=False)
    iteration+=1
    


# In[ ]:

f.close()
'''
os.system("rm -r part-00000")
os.system("rm -r _SUCCESS")
os.system("hdfs dfs -get kmout/*")
'''

# In[91]:


output_file = pd.read_csv('kmout/part-00000',sep='\t',header=None)
output_file = output_file.drop([0,2], axis=1)
clusters = []
for field in output_file[1]:
    field = field.replace(' ','').replace("'",'').replace('[','').replace(']','')
    field = field.split(',')
    clusters.append(field)
predicted_id = [0]*rows
for i,r in enumerate(clusters):
    for values in r:
        predicted_id[int(values)-1] = i
   


# In[ ]:
'''
f = open("part-00000", 'r')
data = f.read()

gen_ids = [0]*total_lines

print(total_lines)
for i in data.strip().split("\n"):
	print(i)
	row = i.strip().split("\t")
	id_list = row[1][1:len(row[1])-1]
	id_list = id_list.replace(' ', '')
	id_list = id_list.replace("'", "")
	id_list = id_list.split(',')
	for k in id_list:
		gen_ids[int(k)-1] = row[0]
'''
jaccard , rand = jaccard_rand(predicted_id,ground_truth)
print('Jaccard value :'+ str(jaccard))
print('Rand Index value :'+ str(rand))
plot_pca(predicted_id,Y)

