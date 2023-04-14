import csv
from statistics import mode

import numpy as np
import pandas as pd
from scipy.spatial import distance

from Dataset import Dataset
from evaluation_module import evaluate

clusters = []
def get_numerical(arr, val, n):
    if val in arr:
        idx = arr.index(val)
    else:
        arr.append(val)
        idx = n
        n += 1
    return idx, n

def get_clusters(y_preds):
    for k in range(y_preds.max() + 1):
        clusters.append(list(np.where(y_preds == k)[0]))
    return clusters
def generate_test_data(dataset: Dataset):
    test_data = pd.read_csv('datasets/corrected', header=None).to_numpy()
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
    test_data = pd.read_csv('datasets/corrected', header=None).to_numpy()
    rows, cols = test_data.shape

    with open('unique_with_labels.csv') as file:
        reader = csv.reader(file)
        feature_1 = next(reader)
        feature_2 = next(reader)
        feature_3 = next(reader)
        labels = next(reader)
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
    clusters = get_clusters(y_preds)
    return evaluate(y_test, y_preds)


if __name__ == "__main__":
    X_test, y_test = generate_test_data_2()
    label_numbers = y_test

    k = [23]    #[7, 15, 23, 31, 45]
    results = []
    for val in k:
        centroids = pd.read_csv(f"kmeans_{val}.csv", header=None, skiprows=[0]).to_numpy()
        results.append([val] + list(evaluate_kmeans(centroids, X_test, y_test)))
    df = pd.DataFrame(data=results, columns=['K', 'Precision', 'Recall', 'F1 Score', 'Conditional Entropy'])
    df.to_csv("kmeans_results.csv")

    dom_label = []
    print(len(clusters))
    for c in range (0, len(clusters)):
        clusters[c] = [label_numbers[x] for x in clusters[c]]
        if len(clusters[c]) == 0:
            dom_label.append(-1)
        else:
            dom_label.append(mode(clusters[c]))

    cluster_purity = 0
    for c in range(0, len(clusters)):

        for idx in clusters[c]:
            if idx == dom_label[c]:
                cluster_purity = cluster_purity + 1

    print(cluster_purity)
    print(dom_label)