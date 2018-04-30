# Example for my blog post at:
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
import functools
import sets
import tensorflow as tf
from data_model import StockDataSet
from config import RNNConfig


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    def __init__(self, data, target, dropout, num_hidden=200, num_layers=3):
        # (batch_size, time_steps, features)
        self.data = data
        # (batch_size, time_steps, output_size)
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        # Create cells with self._num_hidden units and self._num_layers
        # Each cells wrapped by dropout self.dropout
        cells = []
        for _ in range(self._num_layers):
            cell = tf.contrib.rnn.LSTMCell(self._num_hidden)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=1.0 - self.dropout)
            cells.append(cell)
        network = tf.contrib.rnn.MultiRNNCell(cells)
        # self.data -> (batch_size, time_steps, features)
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        # For classification, we might only care about the ouput activation
        # at the last time step. We transpose so that the time axis is first
        # and use tf.gather() for selecting the last frame.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def main():
    # We treat images as sequences of pixel rows.
    config = RNNConfig()
    stock_data = StockDataSet('apple', num_features=config.num_features,
                                 num_classes=config.num_classes,
                                 num_steps=config.num_steps,
                                 test_ratio=0.2,
                                 include_stopwords=config.include_stopwords)
    train_X, train_y, test_X, test_y = stock_data.get_data()
    print('train data shape: {}'.format(train_X.shape))
    print('train target shape: {}'.format(train_y.shape))
    _, num_steps, num_features = train_X.shape
    num_classes = train_y.shape[1]
    # (batch_size, time_steps, features)
    data = tf.placeholder(tf.float32, [None, num_steps, num_features])
    target = tf.placeholder(tf.float32, [None, num_classes])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout, 
                                   num_hidden=config.num_hidden, 
                                   num_layers=config.num_layers)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(config.num_epoch):
        for batch_X, batch_y in stock_data.generate_one_epoch(config.batch_size):
            sess.run(model.optimize, {
                data: batch_X, target: batch_y, dropout: config.dropout})
        train_error = sess.run(model.error, {
            data: train_X, target: train_y, dropout: config.dropout})
        test_error = sess.run(model.error, {
            data: test_X, target: test_y, dropout: config.dropout})
        print('Epoch {:2d} train error: {:4.2f}% test error: {:4.2f}%'.format(epoch + 1, 
                                                                              100 * train_error, 
                                                                              100 * test_error))

if __name__ == '__main__':
    main()