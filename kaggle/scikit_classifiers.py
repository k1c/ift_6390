# scikit_classifier.py
# Isabelle Bouchard
# Carolyne Pelletier
# 2019-11-07
# IFT-6390

import csv
import argparse
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
from sklearn.neural_network import MLPClassifier

_DATA_PATH = pathlib.Path("data/")

_CLASSIFIERS = {
    "MLP": (
        MLPClassifier(hidden_layer_sizes=275, early_stopping=True, validation_fraction=0.005, learning_rate_init=0.0001),
        {
            "hidden_layer_sizes": [225, 250, 275],
            "learning_rate_init": [225, 250, 275],
        }
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [50, 100, 200],
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
    "SVM": (
        SGDClassifier(loss='hinge', alpha=0.001, random_state=42),
        {
            "alpha": [0.0005, 0.001, 0.005, 0.01],
        }
    ),
}

_EMBEDDER = DocumentPoolEmbeddings([WordEmbeddings("glove")], "mean")


_INPUT_OPTIONS = {
    "TFIDF": (False, False),
    "GloveExtraFeatures": (True, True),
    "Glove": (True, False),
}


def read_data(set_):
    return np.load(_DATA_PATH / f"data_{set_}", allow_pickle=True)


def preprocess(X, lem=True, stem=True, embed=True,
               remove_stop_words=True, extra_features=False):
    return [preprocess_line(line_x, lem, stem, embed,
                            remove_stop_words, extra_features) for line_x in X]

def preprocess_line(original_line, lem=True, stem=True, embed=True,
                    remove_stop_words=True, extra_features=False):
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

    # If we have a valid set, it means we are in training mode
    if X_val:
        clf = GridSearchCV(clf, parameters, cv=5, iid=False, n_jobs=-1)

    clf.fit(X_train, y_train)

    # Test mode
    if X_test:
        test_prediction = clf.predict(X_test)
        write_csv(test_prediction)
        return 0., {}

    prediction = clf.predict(X_val)
    accuracy = np.mean(prediction == y_val)
    return accuracy, clf.best_params_


def main(X_train, X_val, X_test, y_train, y_val, clf_name, embed=False):
    clf, params = _CLASSIFIERS[clf_name]
    print(f"Running GS for {clf_name}...")
    accuracy, best_params = run_grid_search(clf, params, X_train, X_val, X_test, y_train, y_val, embed)
    print(f">>> {clf_name} score = {accuracy}")
    print(f"{best_params}")
    return accuracy, best_params


def _read_data(lem, stem, remove_stop_words, extra_features, embed, test=False):
    X_raw, y_raw = read_data(set_="train.pkl")
    X = preprocess(X_raw, lem=lem, stem=stem, remove_stop_words=remove_stop_words,
                   extra_features=extra_features, embed=embed)

    if test:
        X_test = read_data(set_="test.pkl")
        X_test = preprocess(X_test, lem=lem, stem=stem, remove_stop_words=remove_stop_words,
                            extra_features=extra_features, embed=embed)
        return X, None, X_test, y_raw, None

    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y_raw,
                                                      test_size=0.01,
                                                      random_state=42)
    return X_train, X_val, None, y_train, y_val


def _parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--model-name",
        type=str,
        help=f"One of {list(_CLASSIFIERS.keys())}",
    )
    parser.add_argument(
        "--input",
        type=str,
        help=f"One of {list(_INPUT_OPTIONS.keys())}",
    )
    parser.add_argument(
        "--lem",
        action="store_true",
        help="Whether or not to apply lemmatization",
    )
    parser.add_argument(
        "--stem",
        action="store_true",
        help="Whether or not to apply stemming",
    )
    parser.add_argument(
        "--rm-stop-words",
        action="store_true",
        help="Whether or not to remove stop_words",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether or not to run test predictions and to output to csv file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

    args = _parse_args()
    embed, extra_features = _INPUT_OPTIONS[args.input]
    X_train, X_val, X_test, y_train, y_val = _read_data(lem=args.lem, stem=args.stem,
                                                        remove_stop_words=args.rm_stop_words,
                                                        extra_features=extra_features,
                                                        embed=embed, test=args.test)
    main(X_train, X_val, X_test, y_train, y_val, clf_name=args.model_name)
