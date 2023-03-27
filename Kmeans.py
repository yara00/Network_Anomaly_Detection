import math
import time
import csv
import numpy as np
from random import random, randrange

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

from Dataset import Dataset


class Kmeans:
    def __init__(self, data_matrix, k):
        self.data_matrix = data_matrix
        self.k = k
        self.clusters = [[] for i in range(self.k)]

    def __distance(self, current, prev):
        dist_sum = 0
        for i in range(self.k):
            dist = [(a - b) ** 2 for a, b in zip(current[i], prev[i])]
            dist_sum += np.asarray(dist).sum()
        return dist_sum

    def __randomize_centroids(self, k):
        centroids = [[] for i in range(k)]
        for i in range(k):
            rnd = randrange(0, self.data_matrix.shape[0])
            centroids[i].append(self.data_matrix[rnd, :])
        return centroids

    def __assign_clusters(self, centroids, k):
        self.clusters = [[] for j in range(k)]
        similarities = cdist(self.data_matrix, np.asarray(centroids).reshape(k, -1), 'euclidean')
        idx = np.argmin(similarities, axis=1)
        for i in range(self.data_matrix.shape[0]):
            self.clusters[idx[i]].append(i)

    def __update_centroids(self, k):
        centroids = [[] for i in range(k)]
        for i in range(k):
            cluster_sum = np.zeros((1, 41))
            if len(self.clusters[i]) == 0:
                rnd = randrange(0, self.data_matrix.shape[0])
                centroids[i] = self.data_matrix[rnd, :]
            else:
                for j in self.clusters[i]:
                    cluster_sum = self.data_matrix[j, :] + cluster_sum
                centroids[i] = (cluster_sum / len(self.clusters[i]))[0]
        return centroids

    def algorithm(self):
        k = self.k
        iterations = 0
        centroids = self.__randomize_centroids(k)
        prev_centroids = np.zeros(np.array(centroids).shape)
        while (self.__distance(centroids, prev_centroids) != 0) and (iterations < 1000):
            iterations += 1
            self.__assign_clusters(centroids, k)
            prev_centroids = centroids
            centroids = self.__update_centroids(k)
        print("Iterations no: ", iterations)
        return self.clusters, centroids


if __name__ == '__main__':
    k_mat = [7, 15, 23, 31, 45]
    for k in k_mat:
        print("K = ", k)
        start = time.time()
        filename = 'kmeans_' + str(k) + '.csv'
        cluster, centroid = Kmeans(Dataset().generate_data_matrix(), k).algorithm()
        end = time.time()
        print("Elapsed Time: " + str((end - start) / 60) + " mins")
        file = open(filename, 'w', newline='')
        with file:
            write = csv.writer(file)
            write.writerow([k])
            write.writerows(centroid)
