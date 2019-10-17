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
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

DATA_PATH = pathlib.Path("data/")

class BaselineModel:
    def __init__(self):
        self._classes = None

    def train(self, X_train, y_train):
        self._classes = list(np.unique(y_train))

    def predict(self, X_test):
        return [self._classes[random.randint(0, len(self._classes) - 1)] for _ in X_test]


class NaiveBayesModel:
    def __init__(self, vocab, alpha):
        self._vocab = vocab
        self.alpha = alpha
        self._classes = None
        self._class_priors = None
        self._class_cond_densities = None

    def train(self, X_train, y_train):
        self._compute_class_priors(y_train)
        self._compute_class_cond_densities(X_train, y_train)

    def _compute_class_priors(self, y_train):
        classes, counts = np.unique(y_train, return_counts=True)
        self._classes = classes
        self._class_priors = {c: count / len(y_train) for c, count in zip(classes, counts)}

    def _compute_class_cond_densities(self, X, y):
        densities = {}
        dimension = X.shape[0]
        vocab_dimension = len(self._vocab)
        print("dimension",dimension)
        print("vocab_dimension",vocab_dimension)
        for c in self._classes:
            # Get word count (+ 1 is for Laplace smoothing)
            word_count_for_c = np.sum(X[np.where(np.array(y) == c)], axis=0) + self.alpha
            word_count = np.sum(X, axis=0) + (self.alpha * vocab_dimension)
            densities[c] = word_count_for_c / word_count
        self._class_cond_densities = densities

    def _compute_posterior(self, X, c):
        prior = self._class_priors[c]
        p_cond = 1.
        for word in X:
            index = self._vocab.get(word)
            # Word is not in vocab
            if index is None:
                continue
            p_cond *= self._class_cond_densities[c][0, index]
        return p_cond * prior

    def predict(self, X_test):
        y_prediction = []
        for x in X_test:
            best_p = 0.
            pred = None
            for c in self._classes:
                posterior_c = self._compute_posterior(x, c)
                if posterior_c > best_p:
                    best_p = posterior_c
                    pred = c
            y_prediction.append(pred)
        return y_prediction

    # predict on X_val and compare to y_val to get a score
    def get_accuracy(self, X_val, y_val):
        predictions = self.predict(X_val)
        return np.mean(np.asarray(predictions) == np.asarray(y_val))


def build_vocab(X, min_freq=None):
    """
    Inspired from https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    """
    indptr = [0]
    indices = []
    data = []
    vocab = {}
    word_count = {}
    for line in X:
        for word in line:
            if word not in word_count:
                word_count[word] = 1
            word_count[word] += 1
            # Ignore infrequent words
            if word_count[word] < min_freq:
                continue
            index = vocab.setdefault(word, len(vocab))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    matrix = csr_matrix((data, indices, indptr), dtype=int)
    return vocab, matrix

def read_data(set_):
    return np.load(DATA_PATH / f"data_{set_}.pkl", allow_pickle=True)

def preprocess(data, lem=True, stem=True, remove_stop_words=True):
    return [preprocess_line(line, lem, stem, remove_stop_words) for line in data]


def preprocess_line(line, lem=True, stem=True, remove_stop_words=True):
    line = word_tokenize(line)

    if lem:
        lemmatizer = WordNetLemmatizer()
        line = [lemmatizer.lemmatize(word)for word in line]

    if stem:
        stemmer = PorterStemmer()
        line = [stemmer.stem(word) for word in line]

    if remove_stop_words:
        stop_words = set(stopwords.words('english'))
        line = [word for word in line if not word in stop_words]
    return line


def write_csv(y_prediction):
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Id", "Category"])
        for i, y in enumerate(y_prediction):
            writer.writerow([i, y])

def main(X_train, X_val, y_train, y_val, X_test, min_freq, alpha):

    vocab, X_train_sparse = build_vocab(X_train, min_freq=min_freq)
    model = NaiveBayesModel(vocab=vocab, alpha=alpha)
    model.train(X_train_sparse, y_train)
    score = model.get_accuracy(X_val, y_val)

    y_prediction = model.predict(X_test)
    return y_prediction , score


if __name__ == "__main__":
    import nltk
    #nltk.download('wordnet')
    ##nltk.download('stopwords')

    seed = 42

    X_train, y_train = read_data(set_="train")
    X_test = read_data(set_="test")

    X_train = preprocess(X_train, lem=True, stem=True)
    X_test = preprocess(X_test, lem=True, stem=True)

    # split train into train / val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        test_size=0.2,
                                                        random_state=seed)
    best_score = 0.
    best_config = None
    best_predictions = None

    #for min_freq in [0, 1, 5, 10, 100]:
        #for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 2, 100]:
    for min_freq in [0]:
        for alpha in [0.0001]:
            config = f"min_freq {min_freq}, smoothing_param {alpha}"
            print(f">>> {config}")
            y_prediction, score = main(X_train, X_val, y_train, y_val, X_test, min_freq, alpha)
            if score > best_score:
                best_score = score
                best_config = config
                best_predictions = y_prediction

    print(f"Best score: {best_score} \n  Best config: {best_config}")

    write_csv(best_predictions)

