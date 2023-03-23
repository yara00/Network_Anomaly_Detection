import numpy as np
import pandas as pd


class Dataset:
    def __init__(self):
        self.data_matrix = None
        # unique values of each categorical column
        # --> a category's index represents its numeric representation in the data matrix
        self.unique_values1 = []
        self.unique_values2 = []
        self.unique_values3 = []
        self.unique_values41 = []

    def generate_data_matrix(self):
        df = pd.read_csv('kddcup.data.corrected', header=None)
        df.columns = [i for i in range(0, 42)]
        df[1], self.unique_values1 = pd.factorize(df[1])
        df[2], self.unique_values2 = pd.factorize(df[2])
        df[3], self.unique_values3 = pd.factorize(df[3])
        df[41], self.unique_values41 = pd.factorize(df[41])
        self.data_matrix = df.to_numpy()
        return self.data_matrix


if __name__ == '__main__':
    d = Dataset()
    x = d.generate_data_matrix()
    print(x.shape)
    print(x[419984:419995, 1])
