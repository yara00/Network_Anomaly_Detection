import math

import numpy as np
from random import random, randrange
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

from Dataset import Dataset


class Kmeans:
    def __init__(self, k):
        self.k = k
        self.clusters = [[] for i in range(self.k)]
        d = Dataset()
        self.data_matrix, D_test, y_train, y_test = train_test_split(d.generate_data_matrix(),
                                                                     d.labels,
                                                                     train_size=0.005,
                                                                     random_state=42,
                                                                     stratify=d.labels)

    def __randomize_centroids(self, k):
        centroids = [[] for i in range(k)]
        for i in range(k):
            rnd = randrange(0, self.data_matrix.shape[0] + 1)
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
                rnd = randrange(0, self.data_matrix.shape[0] + 1)
                centroids[i] = self.data_matrix[rnd, :]
            else:
                for j in self.clusters[i]:
                    cluster_sum = np.add(self.data_matrix[j, :], cluster_sum)
                centroids[i] = cluster_sum / len(self.clusters[i])

        return centroids

    def algorithm(self):
        k = self.k
        print("K", k)
        ctr = 0
        delta = [1e-3 for i in range(k)]
        centroids = self.__randomize_centroids(k)
        prev_centroids = np.zeros(np.array(centroids).shape)
        while ctr < 10:  # np.allclose(np.array(prev_centroids), np.array(centroids), rtol= 1e-3, atol=1e-3) == False and
            ctr += 1
            print("el kilo ", ctr)
            self.__assign_clusters(centroids, k)
            prev_centroids = centroids
            centroids = self.__update_centroids(k)

        return self.clusters, centroids


if __name__ == '__main__':
    Kmeans(7).algorithm()
