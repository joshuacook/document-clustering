# Databricks notebook source
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
import numpy as np
np.random.seed(42)
cmap = colors.ListedColormap(['blue','green','red'])

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# COMMAND ----------

fig, ax = plt.subplots(1,2,figsize=(20,6))
IRIS = load_iris()
irisDF = pd.DataFrame(IRIS.data)
actual = IRIS.target 

pca = PCA(n_components=4)
iris_scaled = (irisDF - irisDF.mean())/(irisDF.std())
pca_data = pca.fit_transform(iris_scaled)

iris_trans = pd.DataFrame(pca_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc2"]].plot(kind="scatter", x="pc1", y="pc2", c=actual, cmap=cmap, ax=ax[0])

index=[pc+"\n"+str(round(ev, 2)) for pc, ev in zip(["pc1", "pc2", "pc3", "pc4"], pca.explained_variance_ratio_)]
pd.DataFrame(pca.components_, 
             columns=IRIS.feature_names, index=index).plot(kind="bar", cmap="winter", ax=ax[1], rot=0)

display(fig)

# COMMAND ----------

fig, ax = plt.subplots(1,2,figsize=(20,6))

iris_trans = pd.DataFrame(pca_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc2"]].plot(kind="scatter", x="pc1", y="pc2", ax=ax[0])

pd.DataFrame(pca.components_, 
             columns=IRIS.feature_names, 
             index=index).plot(kind="bar", cmap="winter", ax=ax[1], rot=0)

display(fig)

# COMMAND ----------

from sklearn.mixture import GaussianMixture

fig, ax = plt.subplots(1,2,figsize=(20,6))

gmm = GaussianMixture(n_components=2)
gmm.fit(pca_data)
labels = gmm.predict(pca_data)

iris_trans = pd.DataFrame(pca_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc2"]].plot(kind="scatter", x="pc1", y="pc2", c=labels, cmap=cmap, ax=ax[0])

pd.DataFrame(pca.components_, 
             columns=IRIS.feature_names, 
             index=index).plot(kind="bar", cmap="winter", ax=ax[1], rot=0)

display(fig)

# COMMAND ----------

set(labels)

# COMMAND ----------

import matplotlib.cm as cm
import numpy as np

cluster_1 = irisDF[labels == 0]
cluster_2 = irisDF[labels == 1]
coloring_1 = IRIS.target[labels == 0]
coloring_2 = IRIS.target[labels == 1]
cluster_1_scaled = (cluster_1 - cluster_1.mean())/cluster_1.std()
cluster_2_scaled = (cluster_2 - cluster_2.mean())/cluster_2.std()

# COMMAND ----------


fig, ax = plt.subplots(3,2,figsize=(20,18))

pca_1 = PCA(n_components=4)
pca_1_data = pca_1.fit_transform(cluster_1_scaled)

pca_2 = PCA(n_components=4)
pca_2_data = pca_2.fit_transform(cluster_2_scaled)

iris_trans = pd.DataFrame(pca_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc2"]].plot(kind="scatter", x="pc1", y="pc2", c=actual, cmap=cmap, ax=ax[0][0])

pd.DataFrame(pca.components_, 
             columns=IRIS.feature_names, 
             index=index).plot(kind="bar", cmap="winter", ax=ax[0][1], rot=0)


index_1 = [pc+"\n"+str(round(ev, 2)) for pc, ev in zip(["pc1", "pc2", "pc3", "pc4"], pca_1.explained_variance_ratio_)]
iris_trans = pd.DataFrame(pca_1_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc2"]].plot(kind="scatter", x="pc1", y="pc2", c=coloring_1, cmap=cmap, ax=ax[1][0])

pd.DataFrame(pca_1.components_, 
             columns=IRIS.feature_names, 
             index=index_1).plot(kind="bar", cmap="winter", ax=ax[1][1], rot=0)

index_2 = [pc+"\n"+str(round(ev, 2)) for pc, ev in zip(["pc1", "pc2", "pc3", "pc4"], pca_2.explained_variance_ratio_)]
iris_trans = pd.DataFrame(pca_2_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc2"]].plot(kind="scatter", x="pc1", y="pc2", c=coloring_2, cmap=cmap, ax=ax[2][0])

pd.DataFrame(pca_2.components_, 
             columns=IRIS.feature_names, 
             index=index_2).plot(kind="bar", cmap="winter", ax=ax[2][1], rot=0)

ax[0][0].set_xlim(-4, 5)

display(fig)

# COMMAND ----------

fig, ax = plt.subplots(3,2,figsize=(20,18))

pca_1 = PCA(n_components=4)
pca_1_data = pca_1.fit_transform(cluster_1_scaled)

pca_2 = PCA(n_components=4)
pca_2_data = pca_2.fit_transform(cluster_2_scaled)

iris_trans = pd.DataFrame(pca_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc3"]].plot(kind="scatter", x="pc1", y="pc3", c=actual, cmap=cmap, ax=ax[0][0])

pd.DataFrame(pca.components_, 
             columns=IRIS.feature_names, 
             index=index).plot(kind="bar", cmap="winter", ax=ax[0][1], rot=0)


index_1 = [pc+"\n"+str(round(ev, 2)) for pc, ev in zip(["pc1", "pc2", "pc3", "pc4"], pca_1.explained_variance_ratio_)]
iris_trans = pd.DataFrame(pca_1_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc3"]].plot(kind="scatter", x="pc1", y="pc3", c=coloring_1, cmap=cmap, ax=ax[1][0])

pd.DataFrame(pca_1.components_, 
             columns=IRIS.feature_names, 
             index=index_1).plot(kind="bar", cmap="winter", ax=ax[1][1], rot=0)

index_2 = [pc+"\n"+str(round(ev, 2)) for pc, ev in zip(["pc1", "pc2", "pc3", "pc4"], pca_2.explained_variance_ratio_)]
iris_trans = pd.DataFrame(pca_2_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc3"]].plot(kind="scatter", x="pc1", y="pc3", c=coloring_2, cmap=cmap, ax=ax[2][0])

pd.DataFrame(pca_2.components_, 
             columns=IRIS.feature_names, 
             index=index_2).plot(kind="bar", cmap="winter", ax=ax[2][1], rot=0)

ax[0][0].set_xlim(-4, 5)

display(fig)

# COMMAND ----------

from sklearn.mixture import GaussianMixture

fig, ax = plt.subplots(1,3,figsize=(20,6))

gmm_full = GaussianMixture(n_components=3)
gmm_full.fit(pca_data)
labels = gmm_full.predict(pca_data)

gmm_cluster = GaussianMixture(n_components=2)
gmm_cluster.fit(pca_1_data)
labels_1 = gmm_cluster.predict(pca_1_data)

iris_trans = pd.DataFrame(pca_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc2"]].plot(kind="scatter", x="pc1", y="pc2", c=actual, cmap=cmap, ax=ax[0])
iris_trans = pd.DataFrame(pca_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc2"]].plot(kind="scatter", x="pc1", y="pc2", c=labels, cmap=cmap, ax=ax[1])
iris_trans = pd.DataFrame(pca_1_data, columns=["pc1", "pc2", "pc3", "pc4"])
iris_trans[["pc1", "pc2"]].plot(kind="scatter", x="pc1", y="pc2", c=labels_1, cmap=cmap, ax=ax[2])


display(fig)

# COMMAND ----------

actual, labels

# COMMAND ----------

labels_mapped = labels.copy()
labels_mapped = np.where(labels==1, 0, labels_mapped)
labels_mapped = np.where(labels==0, 1, labels_mapped)
actual, labels_mapped

# COMMAND ----------

(actual == labels_mapped).mean()

# COMMAND ----------

labels_clustered = labels_1.copy()
labels_clustered += 1
labels_clustered = np.insert(labels_clustered, 0, [0]*50)
labels_clustered

# COMMAND ----------

(actual == labels_clustered).mean()

# COMMAND ----------

(labels_mapped == labels_clustered).mean()
