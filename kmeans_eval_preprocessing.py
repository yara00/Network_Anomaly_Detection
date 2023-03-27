import csv

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from scipy.spatial import distance

from Dataset import Dataset
from conditional_entropy import get_cond_entropy


def get_numerical(arr, val, n):
    if val in arr:
        idx = arr.index(val)
    else:
        idx = n
        n += 1
    return idx, n


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

    # Evaluation
    precision = precision_score(y_test, y_preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_preds, average='weighted', zero_division=0)
    entropy = get_cond_entropy(y_test, y_preds)

    return precision, recall, f1, entropy


if __name__ == '__main__':
    d = Dataset()
    generate_test_data(d)
