import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Gerar dados de exemplo
np.random.seed(0)
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Definir o número de clusters (grupos)
num_clusters = 2

# Criar um objeto k-means
kmeans = KMeans(n_clusters=num_clusters)

# Ajustar o modelo aos dados
kmeans.fit(X)

# Obter os rótulos de cluster para cada ponto de dados
labels = kmeans.labels_

# Obter as coordenadas dos centróides
centroids = kmeans.cluster_centers_

# Visualizar os resultados
colors = ["g.", "r."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()