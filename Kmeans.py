import math
import csv
import numpy as np
from random import random, randrange
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
        print(dist_sum)
        return dist_sum

    def __randomize_centroids(self, k):
        centroids = [[] for i in range(k)]
        for i in range(k):
            rnd = randrange(0, self.data_matrix.shape[0])
            centroids[i].append(self.data_matrix[rnd, :])
        return centroids

    def __assign_clusters(self, centroids, k):
        print("Assigning clusters..")
        self.clusters = [[] for j in range(k)]
        for i in range(self.data_matrix.shape[0]):
            max_similarity = -1
            idx = -1
            for j in range(len(centroids)):
                rbf = rbf_kernel(self.data_matrix[i, :].reshape(1, -1), np.asarray(centroids[j]).reshape(1, -1))
                if rbf > max_similarity:
                    max_similarity = rbf
                    idx = j
            self.clusters[idx].append(i)

    def __update_centroids(self, k):
        print("Updating centroids..")
        centroids = [[] for i in range(k)]
        for i in range(k):
            print("LEN ", len(self.clusters[i]))
            cluster_sum = np.zeros((1, 41))
            if (len(self.clusters[i]) == 0):
                rnd = randrange(0, self.data_matrix.shape[0])
                centroids[i] = self.data_matrix[rnd, :]
            else:
                for j in self.clusters[i]:
                    cluster_sum = np.add(self.data_matrix[j, :], cluster_sum)
                centroids[i] = cluster_sum / len(self.clusters[i])

        return centroids

    def algorithm(self):
        k = self.k
        print("K", k)
        iterations = 0
        delta = [1e-3 for i in range(k)]
        centroids = self.__randomize_centroids(k)
        prev_centroids = np.zeros(np.array(centroids).shape)
        while (iterations < 10) and (self.__distance(centroids, prev_centroids) > (0.1 * k)):
            iterations += 1
            print("Iteration no: ", iterations)
            self.__assign_clusters(centroids, k)
            prev_centroids = centroids
            centroids = self.__update_centroids(k)

        return self.clusters, centroids


if __name__ == '__main__':
    # 7, 15, 23, 31, 45
    k = 31
    filename = 'kmeans_' + str(k) + '.csv'
    cluster, centroid = Kmeans(Dataset().generate_data_matrix(), k).algorithm()
    file = open(filename, 'w', newline='')

    with file:
        write = csv.writer(file)
        write.writerow([k])
        write.writerows(centroid)