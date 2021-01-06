import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import time

df = pd.read_excel('120_data.xlsx', header=None)
X = df.to_numpy()

start = time.time()
fcm = FCM(n_clusters=3)
fcm.fit(X)


# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)
print('fcm_centers:\n', fcm_centers)
end = time.time()
time_fcm = end - start
print('time_fcm:', time_fcm)

# plot result
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=fcm_labels, cmap='Set1')
ax.scatter(fcm_centers[:, 0], fcm_centers[:, 1], fcm_centers[:, 2], c='b', marker=(5, 1), s=100)
plt.show()