# scikit_classifier.py
# Isabelle Bouchard
# Carolyne Pelletier
# 2019-11-07
# IFT-6390

import csv
import numpy as np
import pathlib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings
from flair.embeddings import WordEmbeddings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

_DATA_PATH = pathlib.Path("data/")

_CLASSIFIERS = {
    "MLP": (
        MLPClassifier(hidden_layer_sizes=250, verbose=1, early_stopping=True),
        {
        }
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [200],
            "max_depth": [15, 25, 35]
        }
    ),
    "Logistic Regression": (
        LogisticRegression(solver="saga", multi_class="multinomial"),
        {
            "penalty": ["l2", "l1"]
        }
    ),
    "Naive Bayes": (
        MultinomialNB(),
        {
            "alpha": [0.15, 0.25, 0.5, 0.65],
        }
    ),
    "SVM_0": (
        LinearSVC(),
        {
        },
    ),
    "SVM_2": (
        SGDClassifier(loss='hinge', alpha=0.001, random_state=42),
        {
        }
    ),
}

_EMBEDDER = DocumentPoolEmbeddings([
    WordEmbeddings("glove"),
], "mean")

def read_data(set_):
    return np.load(_DATA_PATH / f"data_{set_}", allow_pickle=True)

def preprocess_test(X, lem=True, stem=True, embed=True):
    preprocessed_x = []
    print(len(X))
    for i, line_x in enumerate(X):
        p = preprocess_line(line_x, lem, stem, embed)
        if p is None:
            p = np.zeros(4196)
        preprocessed_x.append(p)
        if i % 100 == 0:
            print(i)

    return preprocessed_x



def preprocess(X, y, lem=True, stem=True, embed=True):
    preprocessed_x = []
    preprocessed_y = []
    print(len(X))
    i = 0
    for line_x, line_y in zip(X, y):
        i += 1
        p = preprocess_line(line_x, lem, stem, embed)
        if p is None:
            continue
        if len(p) > 0:
            preprocessed_x.append(p)
            preprocessed_y.append(line_y)
        if i % 100 == 0:
            print(i)

    return preprocessed_x, preprocessed_y

def preprocess_line(original_line, lem=True, stem=True, embed=True,remove_stop_words=True, extra_features=False):
    tokenizer = RegexpTokenizer(r"(?u)\b\w\w+\b")

    # lower sent
    line = original_line.lower()

    line = tokenizer.tokenize(line)

    if extra_features:
        features = _extract_features(original_line, line)

    if lem:
        lemmatizer = WordNetLemmatizer()
        line = [lemmatizer.lemmatize(word) for word in line]

    if stem:
        stemmer = PorterStemmer()
        line = [stemmer.stem(word) for word in line]

    if remove_stop_words:
        stop_words = set(stopwords.words('english'))
        new_line = [word for word in line if not word in stop_words]
        # n_stop_words
        if extra_features:
            features.append(len(line) - len(new_line))
        line = new_line

    line = " ".join(line)

    if embed:
        try:
            sentence = Sentence(line)
            _EMBEDDER.embed(sentence)
            line = sentence.get_embedding().cpu().detach().numpy()
        except Exception:
            return None

    if extra_features:
        # concat features at the end!
        np.concatenate((line, np.asarray(features)))

    return line


def _extract_features(original_line, line):
    # Has a link in it
    has_link = int("http" in original_line)

    # Sentence length
    len_ = len(original_line)

    # Exclamation mark
    ratio_ex = len([c for c in line if c == "!"]) / len_

    # Question mark
    ratio_q = len([c for c in line if c == "?"]) / len_

    # Upper case
    ratio_up = len([c for c in line if c.isupper]) / len_

    return [has_link, len_, ratio_ex, ratio_q, ratio_up]


def write_csv(y_prediction):
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Id", "Category"])
        for i, y in enumerate(y_prediction):
            writer.writerow([i, y])


def run_grid_search(clf, parameters, X_train, X_val, X_test, y_train, y_val, embed=False):
    if not embed:
        clf = Pipeline([
            ('vect', CountVectorizer(stop_words='english', max_features=50000)),
            ('tfidf', TfidfTransformer()),
            ('clf', clf),
        ])
    if X_val:
        clf = GridSearchCV(clf, parameters, cv=2, iid=False, n_jobs=-1)

    clf.fit(X_train, y_train)

    if not X_val:
        test_prediction = clf.predict(X_test)
        write_csv(test_prediction)
        return 0., {}

    prediction = clf.predict(X_val)
    accuracy = np.mean(prediction == y_val)
    return accuracy, clf.best_params_


def main(X_train, X_val, X_test, y_train, y_val, clf_name="SVM_1",
         voting=False, embed=False):

    if isinstance(clf_name, str):
        clf, params = _CLASSIFIERS[clf_name]

    if voting:
        estimators = [(name, _CLASSIFIERS[name][0]) for name in clf_name]
        clf = VotingClassifier(estimators=estimators, voting='hard')
        clf_name = "voting"
        params= {}


    print(f"Running GS for {clf_name}...")
    accuracy, best_params = run_grid_search(clf, params, X_train, X_val, X_test, y_train, y_val, embed)
    print(f">>> {clf_name} score = {accuracy}")
    print(f"{best_params}")
    return accuracy, best_params


def _read_data(save=False, exp_name="", test=False):
    X_test = read_data(set_="test.pkl")
    X_test = preprocess_test(X_test, lem=False, stem=False, embed=False)

    X_raw, y_raw = read_data(set_="train.pkl")
    X, y = preprocess(X_raw, y_raw, lem=False, stem=False, embed=False)

    if test:
        return X, None, X_test, y, None

    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size=0.01,
                                                      random_state=42)
    if save:
        np.save(_DATA_PATH / "data_test_{exp_name}", X_test, allow_pickle=True)
        np.save(_DATA_PATH / "data_train_{exp_name}", (X_train, y_train), allow_pickle=True)
        np.save(_DATA_PATH / "data_val_glove_{exp_name}", (X_val, y_val), allow_pickle=True)

    return X_train, X_val, X_test, y_train, y_val


def _read_preprocessed(exp_name=""):
    X_train, y_train = read_data(set_="train_{exp_name}.npy")
    X_val, y_val = read_data(set_="val_{exp_name}.npy")

    return X_train, X_val, y_train, y_val

if __name__ == "__main__":

    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')


    X_train, X_val, X_test, y_train, y_val = _read_data(test=True)

    model_name = "MLP"
    y_test = main(X_train, X_val, X_test, y_train, y_val, clf_name=model_name, voting=False, embed=False)
