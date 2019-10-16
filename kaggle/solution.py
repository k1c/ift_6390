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
from sklearn.feature_extraction.text import CountVectorizer
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
    def __init__(self):
        self._vocab = None
        self._classes = None
        self._class_priors = None
        self._class_cond_densities = None

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

# TODO model.train should take X_val and y_val and return validation scores
    def train(self, X_train, y_train):
        vocab, X_train_sparse = self._build_vocabulary(X_train)
        self._vocabulary = vocab

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

def preprocess_line(line, analyzer, lem=True, stem=True, remove_stop_words=True):
    # Tokenize
    line = analyzer(line)

    if lem:
        lemmatizer = WordNetLemmatizer()
        line = [lemmatizer.lemmatize(word)for word in line]

    if stem:
        stemmer = PorterStemmer()
        line = [stemmer.stem(word) for word in line]

    if remove_stop_words:
        stop_words = set(stopwords.words('english'))
        data = [word for word in line if not word in stop_words]   #TODO check that data should be data and not line

    return " ".join(line)

def preprocess(data, lem=True, stem=True, remove_stop_words=True, min_df=0.001):

    vectorizer = CountVectorizer(min_df=min_df)
    analyzer = vectorizer.build_analyzer()

    # Preprocess line by line
    data = [preprocess_line(line,
                            analyzer,
                            lem=lem,
                            stem=stem,
                            remove_stop_words=remove_stop_words)
             for line in data]

    # Strings to array
    return vectorizer.fit_transform(data)

def get_data(set_):
    data = read_data(set_)
    return preprocess(data)

def write_csv(y_prediction):
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Id", "Category"])
        for i, y in enumerate(y_prediction):
            writer.writerow([i, y])


def main(seed=42, lem=True, stem=True, remove_stop_words=True, min_df=0.0012):
    X_train, y_train = get_data(set_="train")
    X_test = get_data(set_="test")

    model = NaiveBayesModel()

    # pre_process features
    X = preprocess(X_train, lem, stem, remove_stop_words, min_df)

    # split train into train / val
    X_train, X_val, y_train, y_val = train_test_split(X, y_train,
                                                        test_size=0.2,
                                                        random_state=seed)

    val_score = model.train(X_train, y_train) # TODO model.train should take X_val and y_val and return validation scores

    y_prediction = model.predict(X_test)

    return y_prediction, val_score


if __name__ == "__main__":
    import nltk
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

    seed = 42

    best_score = 0.
    best_config = None
    best_predictions = None

    for lem in [True, False]:
        for stem in [True, False]:
            for remove_stop_words in [True, False]:
                for min_df in [0, 0.0001, 0.001, 0.01, 0.1]:
                    config = f"lem {lem}, stem {stem}, remove_stop_words {remove_stop_words}, min_df {min_df}"
                    print(f">>> {config}")
                    y_prediction, score = main( seed=seed,
                                                lem=lem,
                                                stem=stem,
                                                remove_stop_words=remove_stop_words,
                                                min_df=min_df)
                    if score > best_score:
                        best_score = score
                        best_config = config
                        best_predictions = y_prediction

    print(f"Best score: {best_score} \n  {best_config}")

    write_csv(best_predictions)

