# %matplotlib inline
import pandas as pd #Pandas para usar dataframes
import matplotlib.pyplot as plt #Para graficar
import matplotlib.cm as cm #Para graficar el silhouette
import seaborn as sns #Para graficar
import numpy as np #Para realizar operaciones númericas con matrices y arrays
from sklearn import datasets #sklearn es LA biblioteca de machine learning de python
from sklearn.cluster import KMeans, DBSCAN #Para usar kmeans
from sklearn.preprocessing import StandardScaler #Para estandarizar nuestros datos
from sklearn.metrics import silhouette_samples, silhouette_score #Para el coeficiente de silhouette
from sklearn.cluster import AgglomerativeClustering #Para clustering jerárquico
from sklearn.metrics import pairwise_distances #Para las distancias a pares
from scipy.cluster.hierarchy import dendrogram, cophenet, linkage #Para graficar los dendrogramas y calcular el coeficiente cofenetico
from scipy.cluster import hierarchy #Para graficar los dendrogramas
from scipy.spatial.distance import pdist #Para calcular la distancia con el coeficiente cofenetico
import community as community_louvain #Para louvain
import networkx as nx #Para grafos

path = r'C:\Users\Mateo Sceia\Documents\Facultad Mateo\2do Año\1er Cuatri\Fundamentos de Informatica\datasets\dataset_clustering_teorico.csv'

stock_data = pd.read_csv(path)
stock_data.head()
print(stock_data)
print(stock_data.columns)


print(stock_data.isnull().sum())
print(stock_data.dropna(inplace=True))

print(stock_data.describe())
print(stock_data.info(max_cols=1000))

f = sns.histplot(data = stock_data, x = "2010-01-04", binwidth=0.25, kde = True, color= 'orange')
print(f)
print(stock_data.fillna(stock_data['2010-01-07'].mean(), inplace=True)) 
print(stock_data.fillna(stock_data['2013-10-23'].mean(), inplace=True))


df2 = stock_data.sort_values(by = ["2013-10-29"], ascending = False, ignore_index = True).head(10)
print(df2)

scaler = StandardScaler()
stock_data_normalizado = scaler.fit_transform(stock_data)


Q1 = stock_data['2010-01-04'].quantile(0.07) 
Q2 = stock_data['2010-01-04'].quantile(0.93) 


k = 14  
kmeans = KMeans(n_clusters = k, init="random", n_init=10, max_iter=300, random_state=123457) 
kmeans.fit() 
print(kmeans)