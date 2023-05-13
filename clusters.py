# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:57:36 2023

@author: jksls
"""

#These libraries were used to work with and manipulate the data
import numpy as np
import pandas as pd

#This library was used to draw a graphs
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#These libraries were used for  for analysing and creating ml for clustering technique
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

#This definition creates graphs according the condition if there is any clsuter or no 
def paint(x,y,z,x1,y1,title):
    if (len(z)==0):
        plt.scatter(x,y)
        plt.title(title)
        plt.xlabel(x1)
        plt.ylabel(y1)
        plt.grid(True)
        plt.show()
    else:
        plt.scatter(x,y,color = z)
        plt.title(title)
        plt.xlabel(x1)
        plt.ylabel(y1)
        plt.grid(True)
        plt.show()
  
#This function definies different colors. This can be used for 
def generate_colors(n):
    cmap   = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, n)]
    return colors


#Data importing and deviding it into arrays
df      = pd.read_csv('data.csv',usecols=['Y','Spending(1-100)'])
x_whole = np.array(df)
x       = np.array(df['Y'])
y       = np.array(df['Spending(1-100)'])
  
paint(x, y, [],'Y','Spending(1-100)','Basic graph')

# Scale the variables
scaler    = MinMaxScaler()
df_scaled = scaler.fit_transform(x_whole)

#Defining the Elbow method for the where the max possbile number of clusters according to basic graph
#is 10 (it looks like that there are only 5 clusters and to be sure we will use this method)
SSD = []
for k in range(1,10):
    km = KMeans(k)
    km = km.fit(x_whole)
    SSD.append(km.inertia_)
    
SSD   = np.array(SSD)
SSD_1 = np.empty(8)

for i in range(len(SSD)-2):
    SSD_1[i] = SSD[i]-SSD[i+1]

paint(np.array(range(1,9)),SSD_1,[],'number of clusters','Sum of squeres of distance', 'Elbow graph')
paint(np.array(range(1,10)),SSD,[],'number of clusters','Sum of squeres of distance', 'Elbow graph')

#This part of code is used to define the elbow point automatically
k     = 0
ar_k  = np.empty(9)
for i in range(len(SSD_1)-2):
    if( SSD_1[i] // SSD_1[i+1] == 1 ):
        k      += 1
        ar_k[i] = k
    else:
        k       = 0
        ar_k[i] = 0

elbow_point = np.where( ar_k == max(ar_k) )[0] - max(ar_k) + 2
x_whole1    = df[['Y','Spending(1-100)']]
scaler      = MinMaxScaler()

for i in x_whole1:
        scaler.fit(df[[i]])
        df[i] = scaler.transform(df[[i]])

km             = KMeans(n_clusters = int(elbow_point[0]))
y_predicted    = km.fit_predict(df[['Y','Spending(1-100)']])
df['Clusters'] = y_predicted
n              = int(elbow_point[0])
colors         = generate_colors(n)
result         = np.empty( len(x) , dtype = object )
k              = 0

for i in range(len(y_predicted)):
    k         = y_predicted[i]
    result[i] = colors[k]
    
paint(x, y, result,'Y','Spending(1-100)','Final graph')
