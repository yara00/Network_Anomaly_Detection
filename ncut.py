import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from Dataset import Dataset
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import eigs


class Ncut:
    def normalized_cut(self, D, k):
        A = rbf_kernel(D, gamma=1)
        delta = np.sum(A, axis=1)
        # L = delta - A
        La = np.matmul(np.diag(1 / delta), np.diag(delta) - A)
        eig_values, eig_vectors = eigs(La)
        idx = eig_values.argsort()
        eig_vectors = eig_vectors[:, idx]
        U = eig_vectors[:, 0:k]
        Y = normalize(U.real, axis=1)
        return Y


if __name__ == '__main__':
    d = Dataset()
    D_train, _, y_train, _ = train_test_split(
        d.generate_data_matrix(), d.labels,
        train_size=0.005,
        random_state=42,
        stratify=d.labels)
    ncut = Ncut()
    Y = ncut.normalized_cut(D_train, 23)
    kmeans = KMeans(n_clusters=23, random_state=42, n_init="auto").fit(Y)
    print(kmeans.labels_)
    np.savetxt("labels", y_train)
    np.savetxt("clusters.txt", kmeans.labels_)