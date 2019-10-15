# solution.py
# Carolyne Pelletier
# Isabelle Bouchard
# 2019-10-15
# IFT-6390

import csv
import random
import numpy as np
import pathlib


DATA_PATH = pathlib.Path("data/")


class BaselineModel:
    def __init__(self):
        self._classes = None

    def train(self, X_train, y_train):
        self._classes = list(np.unique(y_train))

    def predict(self, X_test):
        return [self._classes[random.randint(0, len(self._classes) - 1)] for _ in X_test]


def read_data(set_):
    return np.load(DATA_PATH / f"data_{set_}.pkl", allow_pickle=True)

def preprocess(data):
    return data

def get_data(set_):
    data = read_data(set_)
    return preprocess(data)

def write_csv(y_prediction):
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Id", "Category"])
        for i, y in enumerate(y_prediction):
            writer.writerow([i, y])

def main():
    X_train, y_train = get_data(set_="train")
    X_test = get_data(set_="test")

    model = BaselineModel()
    model.train(X_train, y_train)
    y_prediction = model.predict(X_test)
    write_csv(y_prediction)


if __name__ == "__main__":
    main()
