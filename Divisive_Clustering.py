import math

import numpy as np
from sklearn.cluster import KMeans


def divisive_clustering(D, n_clusters):
    index = [x for x in range(len(D))]
    clusters = [index]
    centroids = [0]
    ng = 0
    for j in range(n_clusters):
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(np.array([D[x] for x in clusters[ng]]))
        # centroids = kmeans.cluster_centers_
        cluster1, cluster2 = split(clusters[ng], kmeans.labels_)

        clusters.pop(ng)
        centroids.pop(ng)
        clusters.append(cluster1)
        clusters.append(cluster2)
        centroids.append(kmeans.cluster_centers_[0])
        centroids.append(kmeans.cluster_centers_[1])
        mse = np.zeros(len(clusters))
        indx = 0
        for c in clusters:
            for i in c:
                mse[indx] += math.dist(D[i], centroids[indx]) ** 2
            indx += 1
        ng = np.argmax(mse)
    return clusters


def split(D, labels):
    cluster1 = []
    cluster2 = []
    for i in range(len(labels)):
        if labels[i] == 0:
            cluster1.append(D[i])
        else:
            cluster2.append(D[i])
    return np.array(cluster1), np.array(cluster2)
