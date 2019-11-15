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
import torch
from torch import nn
import math
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import logging

from transformers import BertModel, BertTokenizer
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

DATA_PATH = pathlib.Path("data/")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FREQ = 200

class Bert_MLP():
    def __init__(self,batch_size, train_epochs, optimizer_learning_rate, max_sequence_length):
        self.encoding = BertModel.from_pretrained('bert-base-cased', output_hidden_states=False)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.linear = torch.nn.Linear(768, 275)
        self.classifier = torch.nn.Linear(275, 20) #bert embedding size x number of classifiers
        self.relu = torch.nn.ReLU()
        #self.optimizer = torch.optim.SGD(list(self.linear.parameters()) + list(self.classifier.parameters()), lr=0.0001, momentum=0.9)
        self.optimizer = torch.optim.Adam(list(self.linear.parameters()) + list(self.classifier.parameters()), lr=optimizer_learning_rate)
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.num_train_epochs = train_epochs

    # Bert is a model with absolute position embeddings so it's usually advised
    # to pad the inputs on the right rather than the left.
    # batch is a list of tensors
    def get_zero_pad(self, batch):
        max_length = min(max(s.shape[1] for s in batch), self.max_sequence_length)
        padded_batch = np.zeros((len(batch), max_length))
        for i, s in enumerate(batch):
            padded_batch[i,:s.shape[1]] = s[:max_length]
        return torch.from_numpy(padded_batch).long()

    # Mask to avoid performing attention on padding token indices.
    # Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
    # returns torch.FloatTensor of shape (BATCH_SIZE, sequence_length)
    def get_attention_mask(self, zero_pad_input_ids):
        attention_mask = zero_pad_input_ids.ne(0).float()  # everything in input not equal to 0 will get 1, else 0
        return attention_mask

    def train(self, X_train, labels):
        labels = torch.tensor(labels)

        input_ids = list() #list of torch tensors
        for x in X_train:
            input_ids.append(torch.tensor([self.tokenizer.encode(x, add_special_tokens=True, max_length=self.max_sequence_length)]))

        num_batches = math.ceil(len(input_ids) / self.batch_size)

        criterion = nn.CrossEntropyLoss()

        # right now, using bert as a feature extractor and learning at linear layer level
        # if we want to fine-tune BERT, need to put the encoding parameters + regressor parameters in a list and send it to optimizer

        self.encoding.train()

        #CUDA
        self.encoding.to(DEVICE)
        self.linear.to(DEVICE)
        self.classifier.to(DEVICE)


        for epoch in range(self.num_train_epochs):
            running_loss = 0.0
            for batch_idx in range(num_batches):
                inpud_ids_batch = input_ids[batch_idx * self.batch_size:(batch_idx+1) * self.batch_size]
                zero_pad_input_ids_batch = self.get_zero_pad(inpud_ids_batch)
                attention_mask = self.get_attention_mask(zero_pad_input_ids_batch)

                zero_pad_input_ids_batch = zero_pad_input_ids_batch.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.encoding(input_ids=zero_pad_input_ids_batch, attention_mask=attention_mask) # outputs is a tuple
                last_hidden_states = outputs[0]

                # last_hidden_states is of size (BATCH_SIZE, sequence_length, hidden_size)
                # and I need to bring it down to (BATCH_SIZE, hidden_size) to get a sentence representation (not a word representation)
                # therefore I can use the CLS tokens or I can average over the sequence length (chose the latter)
                sent_emb = last_hidden_states.mean(1) # (BATCH_SIZE, hidden_size)
                out = self.relu(self.linear(sent_emb))
                y_hat = self.classifier(out) #BATCH_SIZE X 20
                labels_batch = labels[batch_idx * self.batch_size:(batch_idx+1) * self.batch_size]
                labels_batch = labels_batch.to(DEVICE)
                tr_loss = criterion(y_hat, labels_batch)
                tr_loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += tr_loss.item()
                if batch_idx % LOG_FREQ == 0:  # print every LOG_FREQ mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, batch_idx + 1, running_loss / LOG_FREQ))
                    running_loss = 0.0


    def predict(self, X_test):
        input_ids = list() #list of torch tensors
        for x in X_test:
            input_ids.append(torch.tensor([self.tokenizer.encode(x, add_special_tokens=True, max_length=self.max_sequence_length)]))

        num_batches = math.ceil(len(input_ids) / self.batch_size)
        predictions = list()
        self.encoding.eval()
        with torch.no_grad():  # using BERT as a feature extractor (freezing BERT's weights and using these to extract features)
            for batch_idx in range(num_batches):
                inpud_ids_batch = input_ids[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
                zero_pad_input_ids_batch = self.get_zero_pad(inpud_ids_batch)
                attention_mask = self.get_attention_mask(zero_pad_input_ids_batch)

                zero_pad_input_ids_batch = zero_pad_input_ids_batch.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)

                #forward
                outputs = self.encoding(input_ids=zero_pad_input_ids_batch, attention_mask=attention_mask) # outputs is a tuple
                last_hidden_states = outputs[0]
                sent_emb = last_hidden_states.mean(1) # (BATCH_SIZE, hidden_size)
                out = self.relu(self.linear(sent_emb))
                y_hat = self.classifier(out) #BATCH_SIZE X 20
                predictions.append(y_hat)
        predictions = torch.cat(predictions, 0)
        return predictions

    # predict on X_val and compare to y_val to get a score
    def get_accuracy(self, X_val, y_val):
        predictions = self.predict(X_val).cpu()
        predictions = predictions.argmax(dim=1).numpy()
        y_val = np.asarray(y_val)
        return accuracy_score(y_val, predictions)

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

def main(is_train, score, X_train, y_train, X_test, lem, stem, remove_stop_words, num_keep, batch_size, train_epochs, optimizer_learning_rate, max_sequence_length):

    #X_train = preprocess(X_train, lem=lem, stem=stem, remove_stop_words=remove_stop_words)
    #X_test = preprocess(X_test, lem=lem, stem=stem, remove_stop_words=remove_stop_words)

    if is_train:
        # split train into train / val
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42) #0.2

    model = Bert_MLP(batch_size, train_epochs, optimizer_learning_rate, max_sequence_length)
    print("Running Training \n")
    model.train(X_train, y_train)

    print("Running Testing \n")
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
    #X_train = X_train[:12]
    #y_train = y_train[:12]

    # convert labels to numbers 0 - 19
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train).tolist()

    X_test = read_data(set_="test")
    #X_test = X_test[:12]

    is_train = True

    best_score = 0.
    score = 0.
    best_config = None
    best_predictions = None
    for lem in [False]:
        for stem in [False]:
            for remove_stop_words in [True]:
                for num_keep in [50000]: # TFIDF keep top 50K
                    for batch_size in [6]: #6
                        for train_epochs in [5]:
                            for optimizer_learning_rate in [5.e-5]: #1e-3
                                for max_sequence_length in [512]:
                                    config = f"lem {lem}, stem {stem}, remove_stop_word {remove_stop_words}, num_keep {num_keep}, batch_size {batch_size}, train_epochs {train_epochs}, optimizer_lr {optimizer_learning_rate}, max_seq_length {max_sequence_length}"
                                    print(f">>> {config}")
                                    y_prediction, score = main(is_train, score, X_train, y_train, X_test, lem, stem,
                                                               remove_stop_words, num_keep, batch_size, train_epochs, optimizer_learning_rate, max_sequence_length)
                                    print("SCORE", score)
                                    if score > best_score:
                                        best_score = score
                                        best_config = config
                                        best_predictions = y_prediction
                                        print("BEST SCORE", best_score)
                                        print("BEST CONFIG", best_config)
