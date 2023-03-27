import csv

from Dataset import Dataset
from Kmeans import Kmeans
from kmeans_eval_preprocessing import generate_test_data, evaluate_kmeans

if __name__ == '__main__':
    d = Dataset()
    X_train = d.generate_data_matrix()
    k_mat = [7, 15, 23, 31, 45]
    n_iter = [10, 10, 10, 5, 5]
    X_test, y_test = generate_test_data(d)
    for i in range(len(k_mat)):
        k = k_mat[i]
        epochs = n_iter[i]
        max_precision = 0
        path = f"outputs/kmeans_{k}.csv"
        print(f"K = {k}")
        for epoch in range(epochs):
            clusters, centroids = Kmeans(X_train, k).algorithm()
            precision, recall, f1 = evaluate_kmeans(centroids, X_test, y_test)
            print(f"epoch={epoch}\tprecision={precision}")
            if precision > max_precision:
                max_precision = precision
                file = open(path, 'w', newline='')
                with file:
                    write = csv.writer(file)
                    write.writerow([k])
                    write.writerows(centroids)
        print(f"max precision = {max_precision}")
