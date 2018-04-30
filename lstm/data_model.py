import numpy as np
import os
import pandas as pd
import random
import time
import json
from sklearn.feature_extraction.text import CountVectorizer

random.seed(time.time())

# TODO: use sparse matrix

class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 num_features=6001,
                 num_classes=2,
                 num_steps=3,
                 test_ratio=0.2,
                 include_stopwords=False):
        self.stock_sym = stock_sym
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.num_features = num_features
        self.num_classes = num_classes
        self.include_stopwords = include_stopwords

        # Read csv file
        raw_df = pd.read_csv(os.path.join("..", "data", stock_sym, "output.csv"))

        # 2d numpy array of shape (# of tweets, 3)
        self.raw_seq = raw_df[['date', 'close', 'tweet']].values
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

    def get_data(self):
        return self.train_X, self.train_y, self.test_X, self.test_y

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    def _prepare_data(self, seq):
        # initialize and fit vectorizer
        if self.include_stopwords:
            stop_words = None
        else:
            stop_words = 'english'
        num_word_features = self.num_features - 1
        vectorizer = CountVectorizer(stop_words=stop_words, max_features=num_word_features, \
                token_pattern='(?u)\\b\\w*[a-zA-Z]\\w*\\b') # ignore numbers
        vectorizer.fit([tweet for _,_,tweet in seq])

        X = np.array([]).reshape(0, self.num_steps, self.num_features)
        y = np.array([]).reshape(0, self.num_classes)
        x_list = np.array([]).reshape(0, self.num_features)
        for i, (date, close, tweet) in enumerate(seq):
            count_vec = vectorizer.transform([tweet]).toarray()
            count_vec = sum(count_vec) # summing the word count for all tweets on date
            count_vec = count_vec / np.linalg.norm(count_vec)
            x = np.array([np.append(count_vec, close)])
            x_list = np.concatenate((x_list, x), axis=0) # append closing price to feature
            if i >= self.num_steps:
                # use (i - self.num_steps) ... (i - 1) tweets to predict the stock price
                # of day (i - 1) to day (i)
                cur_x = x_list[i-self.num_steps:i].reshape(1, self.num_steps, 
                                                        self.num_features)
                X = np.concatenate((X, cur_x), axis=0)
                if seq[i - 1][1] < seq[i][1]:
                    # increase
                    cur_y = np.array([0, 1])
                else:
                    # decrease
                    cur_y = np.array([1, 0])
                cur_y = cur_y.reshape(1, self.num_classes)
                y = np.concatenate((y, cur_y), axis=0)
        

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
        batch_indices = list(range(num_batches))
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y