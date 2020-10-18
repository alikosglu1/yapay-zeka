
# %%
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
data = pd.read_csv("D:/YZSeminerEgitimAgustos/Data/ZiyaretAlisveris.csv")

print(data)


# %%
plt.scatter(data["Ziyaret"], data['AlışVeriş'])
plt.xlabel("Ziyaret")
plt.ylabel("AlışVeriş")
plt.show()


# %%
x = data.copy()
x_scaled = preprocessing.scale(x)
x_scaled


# %%
plt.scatter(x_scaled[:, 0], x_scaled[:, 1])
plt.xlabel("Ziyaret")
plt.ylabel("AlışVeriş")
plt.show()


# %%
kmeans_new = KMeans(n_clusters=4)

# kmeans_new.fit(data)

kmeans_new.fit(x_scaled)

cluster_new = x.copy()

cluster_new["cluster_pred"] = kmeans_new.fit_predict(x_scaled)


# %%
kmeans_new.cluster_centers_


# %%
kmeans_new.cluster_centers_[:, 1]


# %%
kmeans_new.cluster_centers_[:, 0]


# %%
plt.scatter(x_scaled[:, 0], x_scaled[:, 1],
            c=cluster_new["cluster_pred"], cmap="rainbow")
plt.scatter(kmeans_new.cluster_centers_[
            :, 1], kmeans_new.cluster_centers_[:, 0], s=100, c='yellow')
plt.xlabel("Ziyaret")
plt.ylabel("AlışVeriş")
plt.show()


# %%
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel(' Within Cluster Sum of Squares')
plt.show()


# %%
