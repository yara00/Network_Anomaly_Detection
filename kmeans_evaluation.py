import pandas as pd

from kmeans_eval_preprocessing import generate_test_data_2, evaluate_kmeans

if __name__ == "__main__":
    X_test, y_test = generate_test_data_2()
    k = [7, 15, 23, 31, 45]
    results = []
    for val in k:
        centroids = pd.read_csv(f"outputs/kmeans_{val}.csv", header=None, skiprows=[0]).to_numpy()
        results.append([val] + list(evaluate_kmeans(centroids, X_test, y_test)))
    df = pd.DataFrame(data=results, columns=['K', 'Precision', 'Recall', 'F1 Score', 'Conditional Entropy'])
    df.to_csv("Evaluation/kmeans_results.csv")
