# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 09:09:35 2023

@author: rogel
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load your data into a pandas DataFrame
data=pd.read_csv("C:/Users/rogel/Downloads/ATTACKING 500.csv")

# Drop the target variable (if applicable) and any non-numeric columns
X = data.select_dtypes(include=["float64", "int64"])

# Calculate the VIF score for each variable
vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print the results
print(vif)




data.info
data.head()
data.describe()

#Normalizamos los valores
data_normalizada=(data-data.min())/(data.max()-data.min())
data_normalizada.describe()

#Detectar cuantos clusters hay
clusters=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(data)
    clusters.append(kmeans.inertia_)

#Gráfica para ver clusters
plt.plot(range(1,11),clusters)
plt.title("Codo de Jambú")
plt.xlabel("# Clusters")
plt.ylabel("Similitud")
plt.show #Cantidad óptima de clusters es 3

#Ora si a aplicar K-Means
clustering=KMeans(n_clusters=3, max_iter=300)
clustering.fit(data)

data["k_means_clusters"]=clustering.labels_
data.head()

#Tiramos el análisis de componentes principales
pca=PCA(n_components=2)
pca_data=pca.fit_transform(data_normalizada)
pca_data_df=pd.DataFrame(data=pca_data, columns=["Componente_1","Componente_2"])
pca_nombre_data=pd.concat([pca_data_df, data[["k_means_clusters"]]], axis=1)

#Graficamos
fig=plt.figure(figsize=(6,6))

ax=fig.add_subplot(1,1,1)
ax.set_xlabel("Componente 1", fontsize=15)
ax.set_ylabel("Componente 2", fontsize=15)
ax.set_title("Componentes Principales", fontsize=20)

color_theme=np.array(["blue", "red", "yellow"])
ax.scatter(x=pca_nombre_data.Componente_1, y=pca_nombre_data.Componente_2, c=color_theme[pca_nombre_data.k_means_clusters],s=50)

plt.show()

data.to_csv("C:/Users/rogel/Downloads/Clusters con por 90.csv")






