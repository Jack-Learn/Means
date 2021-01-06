from sklearn import cluster, datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import pylab

df = pd.read_excel('120_data.xlsx', header=None)
X = df.to_numpy()

# KMeans 演算法
kmeans_fit = cluster.KMeans(n_clusters = 3).fit(X)

# 印出分群結果
cluster_labels = kmeans_fit.labels_
kmeans_centers = kmeans_fit.cluster_centers_
print('kmeans_centers:\n', kmeans_centers)

# plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_labels, cmap='Set1')
ax.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], kmeans_centers[:, 2], c='b', marker=(5, 1), cmap='Set1', s=100)
plt.show()