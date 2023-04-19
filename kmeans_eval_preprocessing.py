import csv
from statistics import mode

import numpy as np
import pandas as pd
from scipy.spatial import distance

from Dataset import Dataset
from evaluation_module import evaluate, get_precision, get_clustering_summary, get_detected_anomalies

clusters = []
labels_categorical = []

def get_numerical(arr, val, n):
    if val in arr:
        idx = arr.index(val)
    else:
        arr.append(val)
        labels_categorical.append(val)
        idx = n
        n += 1
    return idx, n


def get_clusters(y_preds):
    for k in range(y_preds.max() + 1):
        clusters.append(list(np.where(y_preds == k)[0]))
    return clusters


def generate_test_data(dataset: Dataset):
    test_data = pd.read_csv('corrected', header=None).to_numpy()
    rows, cols = test_data.shape
    feature_1 = list(dataset.unique_values1)
    feature_2 = list(dataset.unique_values2)
    feature_3 = list(dataset.unique_values3)
    labels = list(dataset.labels_unique_values)
    n_feature_1, n_feature_2, n_feature_3, n_labels = len(feature_1), len(feature_2), len(feature_3), len(labels)

    # Map categorical values into numerical values
    for instance in test_data:
        instance[1], n_feature_1 = get_numerical(feature_1, instance[1], n_feature_1)
        instance[2], n_feature_2 = get_numerical(feature_2, instance[2], n_feature_2)
        instance[3], n_feature_3 = get_numerical(feature_3, instance[3], n_feature_3)
        instance[cols - 1], n_labels = get_numerical(labels, instance[cols - 1], n_labels)

    X_test = test_data[:, :cols - 1].astype('float64')
    y_test = test_data[:, cols - 1].astype('float64')
    return X_test, y_test


def generate_test_data_2():
    test_data = pd.read_csv('corrected', header=None).to_numpy()
    rows, cols = test_data.shape

    with open('unique_with_labels.csv') as file:
        reader = csv.reader(file)
        feature_1 = next(reader)
        feature_2 = next(reader)
        feature_3 = next(reader)
        labels = next(reader)
        for l in labels:
            labels_categorical.append(l)
    n_feature_1, n_feature_2, n_feature_3, n_labels = len(feature_1), len(feature_2), len(feature_3), len(labels)

    # Map categorical values into numerical values
    for instance in test_data:
        instance[1], n_feature_1 = get_numerical(feature_1, instance[1], n_feature_1)
        instance[2], n_feature_2 = get_numerical(feature_2, instance[2], n_feature_2)
        instance[3], n_feature_3 = get_numerical(feature_3, instance[3], n_feature_3)
        instance[cols - 1], n_labels = get_numerical(labels, instance[cols - 1], n_labels)

    X_test = test_data[:, :cols - 1].astype('float64')
    y_test = test_data[:, cols - 1].astype('float64')
    return X_test, y_test


def evaluate_kmeans(centroids, X_test, y_test):
    # Determine y_preds
    distance_matrix = distance.cdist(X_test, centroids, 'euclidean')
    y_preds = np.argmin(distance_matrix, axis=1)
    clusters_info = get_clustering_summary(y_test, y_preds)
    global clusters_precision
    precision, clusters_precision = get_precision(clusters_info)
    clusters = get_clusters(y_preds)
    return evaluate(y_test, y_preds)


if __name__ == "__main__":
    X_test, y_test = generate_test_data_2()
    label_numbers = y_test
    print("hey: ", np.count_nonzero(y_test == 5))
    k = [7, 15, 23, 31, 45]
    results = []
    for val in k:
        centroids = pd.read_csv(f"kmeans_{val}.csv", header=None, skiprows=[0]).to_numpy()
        results.append([val] + list(evaluate_kmeans(centroids, X_test, y_test)))
        print("K = ", val)
        print("Detected Anomalies: ")
        get_detected_anomalies(clusters, y_test, clusters_precision, labels_categorical)
        clusters_precision = []
        clusters = []

    df = pd.DataFrame(data=results, columns=['K', 'Precision', 'Recall', 'F1 Score', 'Conditional Entropy'])
    df.to_csv("kmeans_results.csv")



