# solution.py
# Carolyne Pelletier
# Isabelle Bouchard
# 2019-10-15
# IFT-6390

import csv
import random
import numpy as np
import pathlib
from scipy.sparse import csr_matrix

DATA_PATH = pathlib.Path("data/")

class BaselineModel:
    def __init__(self):
        self._classes = None

    def train(self, X_train, y_train):
        self._classes = list(np.unique(y_train))

    def predict(self, X_test):
        return [self._classes[random.randint(0, len(self._classes) - 1)] for _ in X_test]


class NaiveBayesModel:
    def __init__(self):
        self._vocab = None
        self._X_train_sparse = None
        self._y_train = None

    def _build_vocabulary(self, X_train):
        """
        Inspired from https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
        """
        indptr = [0]
        indices = []
        data = []
        vocabulary = {}
        for line in X_train:
            for word in line.split(" "):
                if not word:
                    continue
                index = vocabulary.setdefault(word, len(vocabulary))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
        return vocabulary, csr_matrix((data, indices, indptr), dtype=int)

    def train(self, X_train, y_train):
        vocab, X_train_sparse = self._build_vocabulary(X_train)
        self._vocabulary = vocab
        self._X_train_sparse = X_train_sparse

        self._compute_class_priors(y_train)
        self._compute_class_cond_densities(X_train_sparse, y_train)

    def _compute_class_priors(self, y_train):
        classes, counts = np.unique(y_train, return_counts=True)
        self._classes = classes
        self._class_priors = {c: count / len(y_train) for c, count in zip(classes, counts)}

    def _compute_class_cond_densities(self, X_train_sparse, y_train):
        self._class_cond_densities = {}
        for c in self._classes:
            c_indexes = np.where(np.array(y_train) == c)
            x_cond_y = X_train_sparse[c_indexes]
            word_count = np.sum(x_cond_y, axis=0)
            word_freq = word_count / np.sum(word_count)
            self._class_cond_densities[c] = word_freq

    def _compute_posterior(self, X, c):
        prior = self._class_priors[c]
        p = 1.
        for word in X.split(" "):
            i = self._vocabulary.get(word)
            if i is None:
                continue
            p *= self._class_cond_densities[c][0, i]
        return p * prior

    def predict(self, X_test):
        y_predictions = []
        for x in X_test:
            best_p = 0.
            p_c = None
            for c in self._classes:
                p = self._compute_posterior(x, c)
                if p > best_p:
                    p_c = c
            y_predictions.append(c)
        return y_predictions



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

    model = NaiveBayesModel()
    model.train(X_train, y_train)
    y_prediction = model.predict(X_test)
    write_csv(y_prediction)


if __name__ == "__main__":
    main()
