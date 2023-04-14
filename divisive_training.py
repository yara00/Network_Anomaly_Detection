import numpy as np
import pandas as pd

from Dataset import Dataset
from Divisive_Clustering import divisive_clustering
from evaluation_module import evaluate

if __name__ == "__main__":
    n_iter = 5
    d = Dataset()
    D = d.generate_data_matrix()
    labels = d.labels
    max_precision = 0
    results = []
    for epoch in range(n_iter):
        clusters = divisive_clustering(D, 22)

        C = np.zeros(len(D))
        indx = 0
        for c in clusters:
            for i in c:
                C[i] = indx
            indx += 1

        precision, recall, f1, entropy = evaluate(labels, C)
        print(f"epoch = {epoch}\tprecision = {precision}")
        if precision > max_precision:
            max_precision = precision
            np.savetxt("clustersdiv.txt", C)
            np.savetxt("labels", labels)
            results = [precision, recall, f1, entropy]

    print(f"Precision={results[0]}\nRecall={results[1]}\nF1 Score={results[2]}\nConditional Entropy={results[3]}")
