from statistics import mode

import numpy as np
import pandas as pd
from pandas import Index
from sklearn.metrics import precision_score, recall_score, f1_score


def get_cond_entropy(y_test, y_preds):
    clusters_info = get_clustering_summary(y_test, y_preds)

    # calculate the percentage of each cluster
    total_items_in_cluster = np.sum(clusters_info, axis=1)
    total_items = np.sum(total_items_in_cluster)
    ratio_of_cluster = total_items_in_cluster / total_items

    # calculate the conditional entropy with respect to each cluster
    r = len(clusters_info)
    conditional_entropy_cluster = np.zeros(r)
    for i in range(r):
        for j in range(len(clusters_info[i])):
            if clusters_info[i][j] == 0:
                continue
            ratio_target_cluster = clusters_info[i][j] / total_items_in_cluster[i]
            conditional_entropy_cluster[i] -= ratio_target_cluster * np.log2(ratio_target_cluster)

    # calculate the overall conditional entropy
    conditional_entropy = np.sum(ratio_of_cluster * conditional_entropy_cluster)

    return conditional_entropy


def get_precision(clusters_info: np.ndarray):
    total_items_in_cluster = np.sum(clusters_info, axis=1)
    empty = [i for i in range(len(total_items_in_cluster)) if total_items_in_cluster[i] == 0]
    clusters_info = np.delete(clusters_info, empty, axis=0)
    total_items_in_cluster = np.delete(total_items_in_cluster, empty)
    precision_of_cluster = np.amax(clusters_info, axis=1) / total_items_in_cluster

    total_items = np.sum(total_items_in_cluster)
    ratio_of_cluster = total_items_in_cluster / total_items

    overall_precision = np.sum(ratio_of_cluster * precision_of_cluster)
    return overall_precision, precision_of_cluster


def get_recall(clusters_info: np.ndarray):
    total_items_in_cluster = np.sum(clusters_info, axis=1)
    max_class_in_cluster = np.argmax(clusters_info, axis=1)
    empty = [i for i in range(len(total_items_in_cluster)) if total_items_in_cluster[i] == 0]
    clusters_info = np.delete(clusters_info, empty, axis=0)
    max_class_in_cluster = np.delete(max_class_in_cluster, empty)
    total_items_in_class = np.sum(clusters_info, axis=0)
    recall_of_cluster = np.amax(clusters_info, axis=1) / total_items_in_class[max_class_in_cluster]
    return recall_of_cluster


def get_clustering_summary(y_test, y_preds):
    n, n_clusters, n_labels = len(y_test), int(max(y_preds)) + 1, int(max(y_test)) + 1
    info = np.zeros((n_clusters, n_labels))
    for i in range(n):
        cluster = int(y_preds[i])
        label = int(y_test[i])
        info[cluster][label] += 1
    return info


def get_detected_anomalies(clusters, label_numbers, clusters_precision, labels_categorical):
    dom_label = []
    # majority vote to decide on each cluster's label
    for c in range(0, len(clusters)):
        clusters[c] = [label_numbers[int(x)] for x in clusters[c]]
        if len(clusters[c]) != 0:
            dom_label.append(mode(clusters[c]))
    # eliminate empty clusters
    clusters = [i for i in clusters if i != []]
    detected_anomalies = [0] * len(labels_categorical)
    print("hoooo", (list(clusters_precision)))
    for c in range(0, len(list(clusters_precision))):  # detected += precision * cluster length
        detected_anomalies[int(dom_label[int(c)])] += (clusters_precision[c] * len(clusters[c]))

    # map a cluster number to its categorical name
    for idx in range(0, len(dom_label)):
        dom_label[idx] = labels_categorical[int(dom_label[idx])]

    print("Dominant Label for each cluster: ", dom_label)
    print(labels_categorical)
    print(detected_anomalies)

    df = pd.DataFrame(data=labels_categorical, columns=['Categorical Label'])
    df.insert(1, 'Detected Anomalies', detected_anomalies)
    df.to_csv("kmeans " + str(len(clusters)) + "_anomalies_results.csv")

def get_f1_score(precision, recall):
    r = len(precision)
    f1_score_of_cluster = (2 * precision * recall) / (precision + recall)
    overall_f1_score = np.sum(f1_score_of_cluster) / r
    return overall_f1_score


def evaluate(y_test, y_preds):
    clusters_info = get_clustering_summary(y_test, y_preds)
    precision, cluster_precision = get_precision(clusters_info)
    recall = get_recall(clusters_info)
    f1 = get_f1_score(cluster_precision, recall)
    entropy = get_cond_entropy(y_test, y_preds)

    return precision, recall, f1, entropy


if __name__ == "__main__":
    x = [1, 1, 2, 2, 3]
    print(len(set(x)))
