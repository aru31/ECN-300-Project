import sys
import pandas as pd
import numpy as np

from sklearn import datasets

from generic_summary import LinguisticSummary


# Number of clusters are calculated from the elbow method

def iris_summary():
    """
    :return: iris summary
    """

    iris = datasets.load_iris()
    X = iris.data[:, :]
    y = iris.target
    num_features = len(iris.feature_names)
    summary = LinguisticSummary(X, y, 3, num_features)
    summary.linguistic_summary()


def wheat_seed_summary():
    """
    :return: wheat_seed summary
    """

    features = [
        "Area",
        "Perimeter",
        "Compactness",
        "Kernel.Length",
        "Kernel.Width",
        "Asymmetry.Coeff",
        "Kernel.Groove"
    ]
    y_predict = ["Type"]
    X_df = pd.read_csv('../data/wheatseed.csv', usecols=features)
    y_df = pd.read_csv('../data/wheatseed.csv', usecols=y_predict)
    X1 = np.array(X_df.to_numpy())
    y1 = np.squeeze(np.array(y_df.to_numpy()))
    summary = LinguisticSummary(X1, y1, 3, len(features))
    summary.linguistic_summary()


if __name__ == "__main__":
    data_set = sys.argv[1].lower()
    if data_set == "iris":
        iris_summary()
    elif data_set == "wheat_seed":
        wheat_seed_summary()
