# a1q3.py - COMP550
# Isabelle Bouchard, 2019-09-18

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

_DATA_PATH = pathlib.Path("rt-polaritydata")

_FILENAMES = {
    "pos": "rt-polarity.pos",
    "neg": "rt-polarity.neg"
}

_ENCODING_FORMAT = "latin-1"


_CLASSIFIERS = {
    "Logistic Regression": (LogisticRegression(solver="saga"), {"penalty": ["l1", "l2"]}),
    "Naive Bayes": (MultinomialNB(), {"alpha": [0., 0.25, 0.5, 0.75, 1.]}),
    "SVM": (LinearSVC(dual=False), {"penalty": ["l1", "l2"]}),
    "Random": (DummyClassifier(), {}),
}

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
        data = [word for word in line if not word in stop_words]

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
    return  vectorizer.fit_transform(data)


def plot_confusion_matrix(clf, X_test, y_test, params):
    """
    Inspired from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    y_pred = clf.predict(X_test)

    classes = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    # Normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    # Annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "black" if value > cm.max() / 2. else "white"
            ax.text(j, i, "{:.2f}".format(value), ha="center", va="center", color=color)

def tune_classifier(X, y, name, classifier, parameters, X_test, y_test):
    print(f"Tuning {name} classifier...")
    classifier = GridSearchCV(classifier, parameters, cv=5)
    classifier = classifier.fit(X, y)
    score = classifier.best_score_

    plot_confusion_matrix(classifier, X_test, y_test, parameters)
    print(f"{name}: {score}")
    return score

def main(seed=42, lem=True, stem=True, remove_stop_words=True, min_df=0.0012):
    data = {}
    for label, filename in _FILENAMES.items():
        # Read data from file
        with open(_DATA_PATH / filename, "rb") as f:
            raw_data = f.read()
        # Decode and split lines
        data[label] = raw_data.decode(_ENCODING_FORMAT).split("\n")

    X = preprocess(data["pos"] + data["neg"], lem, stem, remove_stop_words, min_df)
    y = np.array([1] * len(data["pos"]) + [0] * len(data["neg"]))

    # Split into train valid test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=seed)
    best_score = 0.
    best_classifier = None
    for name, (classifier, parameters) in _CLASSIFIERS.items():
        score = tune_classifier(X_train, y_train, name, classifier, parameters, X_test, y_test)
        if score > best_score:
            best_score = score
            best_classifier = name

    return best_score, best_classifier

if __name__ == "__main__":
    import nltk
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

    seed = 42

    best_score = 0.
    best_classifer = None
    best_config = None

    for lem in [True, False]:
        for stem in [True, False]:
            for remove_stop_words in [True, False]:
                for min_df in [0, 0.0001, 0.001, 0.01, 0.1]:
                    config = f"lem {lem}, stem {stem}, remove_stop_words {remove_stop_words}, min_df {min_df}"
                    print(f">>> {config}")
                    score, classifier = main(seed=seed,
                                             lem=lem,
                                             stem=stem,
                                             remove_stop_words=remove_stop_words,
                                             min_df=min_df)
                    if score > best_score:
                        best_score = score
                        best_classifier = classifier
                        best_config = config

    print(f"Best score: {best_score} \n {best_classifier} - {best_config}")
