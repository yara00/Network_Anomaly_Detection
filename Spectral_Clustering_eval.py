from evaluation_module import evaluate

if __name__ == '__main__':
    f = open("outputs/spectral_clusters.txt", "r")
    clusters = f.read().splitlines()
    f = open("outputs/spectral_labels.txt", "r")
    labels = f.read().splitlines()

    for i in range(len(clusters)):
        clusters[i] = float(clusters[i])
        labels[i] = float(labels[i])

    precision, recall, f1, entropy = evaluate(labels, clusters)
    print(f"Precision={precision}\nRecall={recall}\nF1 Score={f1}\nConditional Entropy={entropy}")
