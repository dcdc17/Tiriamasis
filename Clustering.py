import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from constants import tickers, metals


df_all = pd.read_csv('all.csv', index_col=0)
df_all = df_all.iloc[::-1]
df_stocks = df_all[tickers]
df_metals = df_all[metals]

# Clustering --------------
myOGSeries=[df_all[i] for i in df_all.columns]
cluster_count = math.ceil(math.sqrt(len(myOGSeries))) #5
# A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN
mySeries=myOGSeries.copy()
for i in range(len(mySeries)):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(mySeries[i].values.reshape(-1, 1)).flatten()
    mySeries[i] = pd.Series(scaled_data, index=mySeries[i].index)
km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")

labels = km.fit_predict(mySeries)
plot_count = 3
som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Clusters')
row_i = 0
column_j = 0
# For each label there is,
# plots every series with that label
data_time = df_all.index
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
        if (labels[i] == label):
            axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
            cluster.append(mySeries[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(data_time, np.average(np.vstack(cluster), axis=0), c="red")
    axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j))
    column_j += 1
    if column_j % plot_count == 0:
        row_i += 1
        column_j = 0

plt.show()

plot_count = 3
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Clusters')
row_i = 0
column_j = 0
data_time = df_all.index
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
        if (labels[i] == label):
            axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
            cluster.append(mySeries[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(data_time, dtw_barycenter_averaging(np.vstack(cluster)), c="red")
    axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j))
    column_j += 1
    if column_j % plot_count == 0:
        row_i += 1
        column_j = 0

plt.tight_layout()
plt.savefig('clusters.png')
plt.show()


cluster_c = [len(labels[labels==i]) for i in range(cluster_count)] #Cluster distribution
cluster_n = ["Cluster "+str(i) for i in range(cluster_count)]
plt.figure(figsize=(15,5))
plt.title("Cluster Distribution for KMeans")
plt.bar(cluster_n,cluster_c)
plt.show()

fancy_names_for_labels = [f"Cluster {label}" for label in labels]
namesofMySeries = [i.name for i in myOGSeries]
pd.DataFrame(zip(namesofMySeries,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")