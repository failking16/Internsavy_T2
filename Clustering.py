# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:46:29 2023

@author: jksls
"""

#Libararies to work with data
import numpy as np
import pandas as pd

#Libraries that were used for visualizations
from matplotlib import pyplot as plt
plt.style.use('ggplot')
# import matplotlib.cm as cm
# plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle']
# # plt.style.use('fivethirtyeight')

#Libraries that were used for clustering/analysing
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

#This fucntion is used to create graphs according all criteria
def paint(x,y,z):
    if not z:
        pass
    else:
        plt.scatter(x,y)
        plt.title('Basic graph')
        plt.xlabel('Y')
        plt.ylabel('Spending(1-100)')
        plt.grid(True)
        plt.show()

#I am assuming the Spending score as the Y axis, because it has the specific range what is suitable
#for any calculations and findign the influence of other factors on it
#due to high amount of variables and we should cluster data then I will work only with Income
#what allows to work with data more efficiently and we have no aim to analyse data thus 
#such variables as Gender and Age are not required

# Load the data into a pandas dataframe
df = pd.read_csv('data.csv',usecols=['Y','Spending(1-100)'])

x = np.arrya(df['Y'])
y = np.array(df['Spending(1-100)'])

paint(x, y, [])

# Scale the variables
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform([x,y])

print(df_scaled)

# Apply K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42).fit(df_scaled)

# Evaluate the results using silhouette score, WSS, and BSS
labels = kmeans.labels_
silhouette_score = metrics.silhouette_score(df_scaled, labels)
wss = kmeans.inertia_
bss = np.sum((kmeans.cluster_centers_ - df_scaled.mean(axis=0))**2)*df_scaled.shape[0]

# Visualize the clusters using a scatter plot
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=labels)
plt.xlabel('Age')
plt.ylabel('Annual income')
plt.show()






