import numpy as np
import os
import pandas as pd
import random
import time
import json
from sklearn.feature_extraction.text import CountVectorizer

random.seed(time.time())


class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.1,
                 normalized=True):
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.normalized = normalized

        # Read csv file
        raw_df = pd.read_csv(os.path.join("..", "data", stock_sym, "output.csv"))
        # Read vocab file
        # with open(os.path.join("..", "data", stock_sym, "vocab.json"), 'r') as vocab_file:
        #     self.word2idx = json.load(vocab_file)
        # self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # 2d numpy array of shape (# of tweets, 3)
        self.raw_seq = raw_df[['date', 'close', 'tweet']].values

        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    def _prepare_data(self, seq):
        # list of [date, close, [tweets...]]
        # data_list = []
        # for date, close, tweet in seq:
        #     if not data_list or date != data_list[-1][0]:
        #         data_list.append((date, close, []))
        #     data_list[-1][2].append(tweet)
        # initialize and fit vectorizer
        stop_words = None # 'english'
        num_word_features = 5066
        self.num_features = num_word_features + 1 # add one closing price as feature
        vectorizer = CountVectorizer(stop_words=stop_words, max_features=num_word_features, \
                token_pattern='(?u)\\b\\w*[a-zA-Z]\\w*\\b') # ignore numbers
        vectorizer.fit([tweet for _,_,tweet in seq])

        # split into groups of num_steps
        # X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        # y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        X = np.array([]).reshape(0, self.num_features)
        y = np.array([])
        # for i, (date, close, tweets) in enumerate(data_list[:-1]):
        #     count_vec = vectorizer.transform(tweets).toarray()
        #     count_vec = sum(count_vec) # summing the word count for all tweets on date
        #     count_vec = count_vec / np.linalg.norm(count_vec)
        #     x = np.array([np.append(count_vec, close)])
        #     X = np.concatenate((X, x), axis=0) # append closing price to feature
        #     # TODO: fix output
        #     y = np.append(y, data_list[i + 1][1]) # append the next day's closing price
        for i, (date, close, tweet) in enumerate(seq[:-1]):
            count_vec = vectorizer.transform([tweet]).toarray()
            count_vec = sum(count_vec) # summing the word count for all tweets on date
            count_vec = count_vec / np.linalg.norm(count_vec)
            x = np.array([np.append(count_vec, close)])
            X = np.concatenate((X, x), axis=0) # append closing price to feature
            if seq[i + 1][1] - seq[i][1] > 0:
                y = np.append(y, 1)
            else:
                y = np.append(y, 0)

        print('X: {}'.format(X.shape))
        print('y: {}'.format(y.shape))
        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        
        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = range(num_batches)
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
   
            yield batch_X, batch_y