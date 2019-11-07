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
import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

DATA_PATH = pathlib.Path("data/")

_CLASSIFIERS = {
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, random_state=42),
        {
            'vect__min_df': (2, 10, 20),
            'vect__max_df': (100, 250, 500),
            'clf__max_depth': (50, 100, 125)
        }
    ),
    "Logistic Regression": (
        LogisticRegression(solver="saga", multi_class="multinomial"),
        {
            "clf__penalty": ["l1", "l2"],
            'vect__min_df': (2, 10, 20),
            'vect__max_df': (100, 250, 500),

        }
    ),
    "Naive Bayes": (
        MultinomialNB(),
        {
            "clf__alpha": [0., 0.25, 0.5, 0.75, 1.],
            'vect__min_df': (2, 10, 20),
            'vect__max_df': (100, 250, 500),

        }
    ),
    "SVM_0": (
        LinearSVC(),
        {
            'vect__min_df': (2, 10, 20),
            'vect__max_df': (100, 250, 500),
        },
    ),
    "SVM_1": (
        SVC(),
        {
            'vect__min_df': (2, 10, 20),
            'vect__max_df': (100, 250, 500),
        },
    ),
    "SVM_2": (
        SGDClassifier(loss='hinge', random_state=42),
        {
            'vect__min_df': (2, 10, 20),
            'vect__max_df': (100, 250, 500),
            'clf__alpha': (1e-2, 1e-3),
        }
    ),
}


def read_data(set_):
    return np.load(DATA_PATH / f"data_{set_}.pkl", allow_pickle=True)


def preprocess(data, lem=True, stem=True):
    return [preprocess_line(line, lem, stem) for line in data]


def preprocess_line(line, lem=True, stem=True):
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

    return " ".join(line)


def write_csv(y_prediction):
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Id", "Category"])
        for i, y in enumerate(y_prediction):
            writer.writerow([i, y])


def run_grid_search(clf, parameters, X_train, X_val, y_train, y_val):
    text_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', clf),
    ])

    gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)

    gs_clf.fit(X_train, y_train)

    prediction = gs_clf.predict(X_val)
    accuracy = np.mean(prediction == y_val)

    return accuracy, gs_clf.best_params_


def main(X_train, X_val, y_train, y_val, clf_name="SVM_1", bagging=False, boosting=False):
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


    print(f"Running GS for {clf_name}...")
    accuracy, best_params = run_grid_search(clf, params, X_train, X_val, y_train, y_val)
    print(f">>> {clf_name} score = {accuracy}")
    print(f"{best_params}")
    return accuracy, best_params



if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('punkt')

    # Read data
    X_raw, y_raw = read_data(set_="train")

    for lem in [True, False]:
        print(f"LEM {lem}")
        for stem in [True, False]:
            print(f"STEM {lem}")
            # Preprocess data
            X = preprocess(X_raw, lem=lem, stem=stem)
            X_train, X_val, y_train, y_val = train_test_split(X,
                                                              y_raw,
                                                              test_size=0.2,
                                                              random_state=42)
            for model in _CLASSIFIERS.keys():
                main(X_train, X_val, y_train, y_val, model)


# TODO:
# - Voting classifier
#   >>> from sklearn.ensemble import VotingClassifier
#   >>> model1 = LogisticRegression(random_state=1)
#   >>> model2 = tree.DecisionTreeClassifier(random_state=1)
#   >>> model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
