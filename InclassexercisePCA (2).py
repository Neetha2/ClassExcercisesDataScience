#!/usr/bin/env python
# coding: utf-8

# In[14]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA results
principalDf = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
principalDf['Species'] = y

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = [0, 1, 2]
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indicesToKeep = principalDf['Species'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'Principal Component 1'],
               principalDf.loc[indicesToKeep, 'Principal Component 2'],
               c=color, s=50)

ax.legend(target_names)
ax.grid()

