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
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import array

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
        print("dimension", dimension)
        print("vocab_dimension", vocab_dimension)
        for c in self._classes:
            # Get word count (+ alpha is for Laplace smoothing)
            word_count_for_c = np.sum(X[np.where(np.array(y) == c)], axis=0) + self.alpha
            word_count = np.sum(word_count_for_c) + (self.alpha * vocab_dimension)
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


# source: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
# TF IDF is TF multiplied by IDF
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


# source: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
#The log of the number of documents divided by the number of documents that contain the word
def computeIDF(documents):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


# source: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
#The number of times a word appears in a document divded by the total number of words in the document.
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


# source: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
def build_vocab_tfidf(X_train, subreddits, num_keep):  # X_train is a list of strings
    tokenizer = RegexpTokenizer(r"(?u)\b\w\w+\b")

    bag_of_words = list()
    for subreddit in subreddits:
        bag_of_words.append(tokenizer.tokenize(subreddit))

    uniqueWords = set().union(*bag_of_words)

    num_of_words = list()
    for bow in bag_of_words:
        numOfWords = dict.fromkeys(uniqueWords, 0)
        for word in bow:
            numOfWords[word] += 1
        num_of_words.append(numOfWords)

    # Calculate Term Frequency (TF) per class
    tf = list()
    for i in range(len(num_of_words)):
        tf.append(computeTF(num_of_words[i], bag_of_words[i]))

    # Calculate Inverse Data Frequency (IDF)
    idfs = computeIDF(num_of_words)

    # Calculate TF IDF
    tfidf = list()
    for i in range(len(tf)):
        tfidf.append(computeTFIDF(tf[i], idfs))

    df = pd.DataFrame(tfidf)

    # sort each row (aka each class) by descending order and keep the top num_keep
    total_vocab = []
    num_index_rows = len(df.index.values.tolist())
    for i in range(num_index_rows):
        total_vocab.extend(df.sort_values(by=i, axis=1, ascending=False).columns.values.tolist()[:num_keep])

    total_vocab = list(set(total_vocab))

    # We built a confusion matrix and saw that words in 'conspiracy' subreddit are often misclassified
    # This may be due to the fact that these words are shared amongst other classes at a high frequency
    # Therefore perform TFIDF and remove top-X number of words sorted by 'conspiracy' class (index 13)
    #conspiracy_df = df.sort_values(by=13, axis=1, ascending=False, inplace=False)  # sorted by top conspiracy words
    #conspiracy_df.drop(conspiracy_df.columns[:num_keep].values, axis=1, inplace=True)

    X_train = [' '.join(x) for x in X_train]  # convert list of list of strings to list of strings
    vocab = {w: i for i, w in enumerate(total_vocab)}
    x_train_sparse = make_sparse_matrix(X_train, vocab)
    return vocab, x_train_sparse


# source: https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/feature_extraction/text.py#L952
def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))


# inspired by source: https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/feature_extraction/text.py#L952
def make_sparse_matrix(sentences, vocab):
    """Create sparse feature matrix
    """
    vocabulary = vocab

    tokenizer = RegexpTokenizer(r'\w+')

    j_indices = []
    indptr = []

    values = _make_int_array()
    indptr.append(0)
    for doc in sentences:  # looping over sentences
        feature_counter = {}
        for feature in tokenizer.tokenize(doc):
            try:
                feature_idx = vocabulary[feature]
                if feature_idx not in feature_counter:
                    feature_counter[feature_idx] = 1
                else:
                    feature_counter[feature_idx] += 1
            except KeyError:
                # Ignore out-of-vocabulary items for fixed_vocab=True
                continue

        j_indices.extend(feature_counter.keys())
        values.extend(feature_counter.values())
        indptr.append(len(j_indices))

    indices_dtype = np.int32
    j_indices = np.asarray(j_indices, dtype=indices_dtype)
    indptr = np.asarray(indptr, dtype=indices_dtype)
    values = np.frombuffer(values, dtype=np.intc)

    X = csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, len(vocabulary)), dtype=int)
    X.sort_indices()
    return X


def read_data(set_):
    return np.load(DATA_PATH / f"data_{set_}.pkl", allow_pickle=True)


def preprocess(data, lem=True, stem=True, remove_stop_words=True):
    return [preprocess_line(line, lem, stem, remove_stop_words) for line in data]


def preprocess_line(line, lem=True, stem=True, remove_stop_words=True):
    tokenizer = RegexpTokenizer(r"(?u)\b\w\w+\b")

    # lower sent
    line = line.lower()

    line = tokenizer.tokenize(line)

    if remove_stop_words:
        stop_words = set(stopwords.words('english'))
        line = [word for word in line if not word in stop_words]

    if lem:
        lemmatizer = WordNetLemmatizer()
        line = [lemmatizer.lemmatize(word) for word in line]

    if stem:
        stemmer = PorterStemmer()
        line = [stemmer.stem(word) for word in line]

    return line


def write_csv(y_prediction):
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Id", "Category"])
        for i, y in enumerate(y_prediction):
            writer.writerow([i, y])


def get_text(df, label):
    return df[df['labels'] == label]['text']


# subreddits is a list of length 20 (20 classes)
# each element is a string and represents all the text from a single subreddit
def get_subreddits(df, lem, stem, remove_stop_words):
    labels = df.labels.unique().tolist()
    df_text_per_class = list()  # text_per_class is a list of strings
    for label in labels:
        df_text_per_class.append(get_text(df, label))

    text_per_class = list()
    for i in range(len(df_text_per_class)):
        text_per_class.append(df_text_per_class[i].tolist())

    # clean vocab using same params as training data
    text_per_class = [preprocess(text_per_class[i], lem, stem, remove_stop_words) for i in range(len(text_per_class))]

    subreddits = list()
    for i in range(len(text_per_class)):
        X = [' '.join(x) for x in text_per_class[i]]  # convert list of list of strings to list of strings
        X = " ".join(X)
        subreddits.append(X)
    return subreddits


def main(is_train, score, X_train, y_train, X_test, lem, stem, remove_stop_words, alpha, num_keep):
    # build pandas dataframe to gather text from subreddits together
    df_X = pd.DataFrame(X_train, columns=['text'])
    df_Y = pd.DataFrame(y_train, columns=['labels'])
    df = pd.concat([df_X, df_Y], axis=1)

    X_train = preprocess(X_train, lem=lem, stem=stem, remove_stop_words=remove_stop_words)
    X_test = preprocess(X_test, lem=lem, stem=stem, remove_stop_words=remove_stop_words)

    if is_train:
        # split train into train / val
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    subreddits = get_subreddits(df, lem, stem, remove_stop_words)

    vocab, X_train_sparse = build_vocab_tfidf(X_train, subreddits, num_keep)

    model = NaiveBayesModel(vocab=vocab, alpha=alpha)
    model.train(X_train_sparse, y_train)
    y_prediction = model.predict(X_test)

    if is_train:
        score = model.get_accuracy(X_val, y_val)
    else:
        write_csv(y_prediction)

    return y_prediction, score


if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')

    X_train, y_train = read_data(set_="train")
    X_test = read_data(set_="test")

    is_train = False

    best_score = 0.
    score = 0.
    best_config = None
    best_predictions = None
    for lem in [False]:  # HP_Search params: [True, False]
        for stem in [True]:  # HP_Search params: [True, False]
            for remove_stop_words in [True]:  # HP_Search params: [True, False]
                for alpha in [0.1]:  # HP_Search params: [0.01, 0.05, 0.1, 0.15, 0.25, 0.5]
                    for num_keep in [55350]:  # HP_Search params: [40000,50000,540000,55000,55350]
                        config = f"smoothing_param {alpha}, lem {lem}, stem {stem}, remove_stop_word {remove_stop_words}, num_keep {num_keep}"
                        print(f">>> {config}")
                        y_prediction, score = main(is_train, score, X_train, y_train, X_test, lem, stem,
                                                   remove_stop_words, alpha, num_keep)
                        print("SCORE", score)
                        if score > best_score:
                            best_score = score
                            best_config = config
                            best_predictions = y_prediction
                            print("BEST SCORE", best_score)
                            print("BEST CONFIG", best_config)

