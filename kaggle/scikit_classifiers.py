# random_forest.py
# Isabelle Bouchard
# Carolyne Pelletier
# 2019-11-07
# IFT-6390

import csv
import numpy as np
import pathlib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings
from flair.embeddings import FlairEmbeddings
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
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

DATA_PATH = pathlib.Path("data/")

_CLASSIFIERS = {
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, max_depth=150, random_state=42),
        {
        }
    ),
    "Logistic Regression": (
        LogisticRegression(solver="saga", multi_class="multinomial"),
        {
            "clf__penalty": ["l2", "l1"]

        }
    ),
    "Naive Bayes": (
        MultinomialNB(),
        {
            "clf__alpha": [0.15, 0, 2, 0.25, 0.3, 0., 35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
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
    FlairEmbeddings("en-forward"),
    FlairEmbeddings("en-backward"),
    WordEmbeddings("glove"),
], "mean")

def read_data(set_):
    return np.load(DATA_PATH / f"data_{set_}.pkl", allow_pickle=True)


def preprocess(X, y, lem=True, stem=True, embed=True):
    preprocessed_x = []
    preprocessed_y = []
    for line_x, line_y in zip(X, y):
        p = preprocess_line(line_x, lem, stem, embed)
        if len(p) == 0:
            preprocessed_x.append(p)
            preprocessed_y.append(line_y)
    return preprocessed_x, preprocessed_y

def preprocess_line(line, lem=True, stem=True, embed=True):
    tokenizer = RegexpTokenizer(r"(?u)\b\w\w+\b")

    # lower sent
    line = line.lower()

    line = tokenizer.tokenize(line)

    if lem:
        lemmatizer = WordNetLemmatizer()
        line = [lemmatizer.lemmatize(word) for word in line]

    if stem:
        stemmer = PorterStemmer()
        line = [stemmer.stem(word) for word in line]

    line = " ".join(line)

    if embed:
        try:
            sentence = Sentence(line)
            _EMBEDDER.embed(sentence)
            line = sentence.get_embedding().detach().numpy()
        except Exception:
            return None

    return line


def write_csv(y_prediction):
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Id", "Category"])
        for i, y in enumerate(y_prediction):
            writer.writerow([i, y])


def run_grid_search(clf, parameters, X_train, X_val, y_train, y_val, embed=False):
    if embed:
        text_clf = clf
    else:
        text_clf = Pipeline([
            ('vect', CountVectorizer(stop_words='english', min_df=2, max_df=500)),
            ('tfidf', TfidfTransformer()),
            ('clf', clf),
        ])

    gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)

    gs_clf.fit(X_train, y_train)

    prediction = gs_clf.predict(X_val)
    accuracy = np.mean(prediction == y_val)

    return accuracy, gs_clf.best_params_


def main(X_train, X_val, y_train, y_val, clf_name="SVM_1",
         bagging=False, boosting=False, voting=False, embed=False):

    if isinstance(clf_name, str):
        clf, params = _CLASSIFIERS[clf_name]

    if clf_name == "RandomForest" and (boosting or bagging):
        raise ValueError(f"no boost or bagg for {clf_name}")


    if bagging:
        clf = BaggingClassifier(clf)
        clf_name += " bagging"
        params = {}

    if boosting:
        clf = AdaBoostClassifier(clf, algorithm='SAMME')
        clf_name += " boost"
        params = {}

    if voting:
        estimators = [(name, _CLASSIFIERS[name][0]) for name in clf_name]
        clf = VotingClassifier(estimators=estimators, voting='hard')
        clf_name = "voting"
        params= {}


    print(f"Running GS for {clf_name}...")
    accuracy, best_params = run_grid_search(clf, params, X_train, X_val, y_train, y_val, embed)
    print(f">>> {clf_name} score = {accuracy}")
    print(f"{best_params}")
    return accuracy, best_params



if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('punkt')

    # Read data
    X_raw, y_raw = read_data(set_="train")

    for lem in [False]:
        print(f"LEM {lem}")
        for stem in [False]:
            print(f"STEM {stem}")
            # Preprocess data
            X, y = preprocess(X_raw, y_raw, lem=lem, stem=stem, embed=True)
            X_train, X_val, y_train, y_val = train_test_split(X,
                                                              y,
                                                              test_size=0.2,
                                                              random_state=42)
            for model_name in _CLASSIFIERS.keys():
                main(X_train, X_val, y_train, y_val, clf_name=model_name, embed=True)
