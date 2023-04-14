import numpy as np
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
    precision_of_cluster = np.amax(clusters_info, axis=1) / total_items_in_cluster

    total_items = np.sum(total_items_in_cluster)
    ratio_of_cluster = total_items_in_cluster / total_items

    overall_precision = np.sum(ratio_of_cluster * precision_of_cluster)

    return overall_precision, precision_of_cluster


def get_recall(clusters_info: np.ndarray):
    max_class_in_cluster = np.argmax(clusters_info, axis=1)
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
