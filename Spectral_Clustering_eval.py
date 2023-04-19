import numpy as np

from Dataset import Dataset
from evaluation_module import evaluate, get_detected_anomalies, get_precision, get_clustering_summary

if __name__ == '__main__':
    f = open("outputs/spectral_clusters.txt", "r")
    clusters = f.read().splitlines()
    f = open("outputs/spectral_labels.txt", "r")
    labels = f.read().splitlines()

    for i in range(len(clusters)):
        clusters[i] = int(float(clusters[i]))
        labels[i] = int(float(labels[i]))

    clusters_group = []

    for k in range(0, max(clusters) + 1):
        clusters_group.append(list(np.where(np.asarray(clusters) == k)[0]))

    precision, recall, f1, entropy = evaluate(labels, clusters)
    print(f"Precision={precision}\nRecall={recall}\nF1 Score={f1}\nConditional Entropy={entropy}")
    precision, clusters_precision = get_precision(get_clustering_summary(labels, clusters))
    d = Dataset()
    d.generate_data_matrix()
    get_detected_anomalies(clusters_group, labels, clusters_precision, d.labels_unique_values)
