import numpy as np


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


def get_clustering_summary(y_test, y_preds):
    n, n_clusters, n_labels = len(y_test), int(max(y_preds))+1, int(max(y_test))+1
    info = np.zeros((n_clusters, n_labels))
    for i in range(n):
        cluster = int(y_preds[i])
        label = int(y_test[i])
        info[cluster][label] += 1
    return info


if __name__ == "__main__":
    x = [1, 1, 2, 2, 3]
    print(len(set(x)))