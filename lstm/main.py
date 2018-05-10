# Example for my blog post at:
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
import functools
import tensorflow as tf
from data_model import StockDataSet
from config import RNNConfig
from functools import reduce

# https://danijar.com/structuring-your-tensorflow-models/
def define_scope(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

# convert price difference to UP, DOWN, STAY according to stay_percent
def price_to_tag(price_diff, stay_percent=0.005):
    if abs(price_diff) <= stay_percent:
        return 'STAY'
    elif price_diff > 0:
        return 'UP'
    else:
        return 'DOWN'

def tf_price_to_tag(price_diff, stay_percent=0.005):
    return tf.cond(tf.less_equal(tf.abs(price_diff), stay_percent),
                   lambda: 'STAY',
                   lambda: tf.cond(tf.greater(price_diff, 0),
                                   lambda: 'UP',
                                   lambda: 'DOWN'))

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

    @define_scope
    def prediction(self):
        # Recurrent network.
        # Create cells with self._num_hidden units and self._num_layers
        # Each cells wrapped by dropout self.dropout
        with tf.name_scope('model'):
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
        with tf.name_scope('weight_and_bias'):
            weight, bias = self._weight_and_bias(
                self._num_hidden, int(self.target.get_shape()[1]))
        # Softmax layer.
        with tf.name_scope('softmax'):
            prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @define_scope
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @define_scope
    def optimize(self):
        learning_rate = 0.0003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

class SequenceRegression:
    def __init__(self, data, target, dropout, stay_percent, num_hidden=200, num_layers=3):
        # (batch_size, time_steps, features)
        self.data = data
        # (batch_size, time_steps, output_size)
        self.target = target
        self.dropout = dropout
        self.stay_percent = stay_percent
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @define_scope
    def prediction(self):
        # Recurrent network.
        # Create cells with self._num_hidden units and self._num_layers
        # Each cells wrapped by dropout self.dropout
        with tf.name_scope('model'):
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
        with tf.name_scope('weight_and_bias'):
            self.weight, self.bias = self._weight_and_bias(
                self._num_hidden, int(self.target.get_shape()[1]))
        with tf.name_scope('prediction'):
            prediction = tf.matmul(last, self.weight) + self.bias
        return prediction

    @define_scope
    def cost(self):
        beta = 0.01
        weight_regularizer = tf.nn.l2_loss(self.weight)
        bias_regularizer = tf.nn.l2_loss(self.bias)
        mean_squared_loss = tf.reduce_mean(tf.squared_difference(self.prediction, self.target))
        loss = tf.reduce_mean(mean_squared_loss + 
                            beta * weight_regularizer + 
                            beta * bias_regularizer)
        return loss

    @define_scope
    def optimize(self):
        learning_rate = 0.0003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(tf.map_fn(tf_price_to_tag, 
                                          tf.reshape(self.target, [-1]), 
                                          dtype=tf.string),
                                tf.map_fn(tf_price_to_tag, 
                                          tf.reshape(self.prediction, [-1]),
                                          dtype=tf.string))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.1)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

def main():
    tf.reset_default_graph()
    logs_path = './tensorboard_output/model_' + datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
    os.mkdir(logs_path)
    config = RNNConfig()
    stock_data = StockDataSet(config.company, num_features=config.num_features,
                                 num_classes=config.num_classes,
                                 num_steps=config.num_steps,
                                 test_ratio=0.2,
                                 include_stopwords=config.include_stopwords,
                                 stay_percent=config.stay_percent)
    train_X, train_y, test_X, test_y = stock_data.get_data()
    print('train data shape: {}'.format(train_X.shape))
    print('train target shape: {}'.format(train_y.shape))
    _, num_steps, num_features = train_X.shape
    num_classes = train_y.shape[1]
    with tf.name_scope('input'):
        # (batch_size, time_steps, features)
        data = tf.placeholder(tf.float32, [None, num_steps, num_features])
        # (batch_size, num_classes)
        target = tf.placeholder(tf.float32, [None, num_classes])
    with tf.name_scope('dropout'):
        dropout = tf.placeholder(tf.float32)
    if config.num_classes == 1:
        model = SequenceRegression(data, target, dropout, config.stay_percent,
                                    num_hidden=config.num_hidden,
                                    num_layers=config.num_layers)
    else:
        model = SequenceClassification(data, target, dropout, 
                                    num_hidden=config.num_hidden, 
                                    num_layers=config.num_layers)
    # create a summary for our cost and error
    tf.summary.scalar("cost", model.cost)
    tf.summary.scalar("error", model.error)
    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # create log writer object
        writer_train = tf.summary.FileWriter(logs_path + 'plot_train') #, graph=tf.get_default_graph(), graph=tf.get_default_graph())
        writer_val = tf.summary.FileWriter(logs_path + 'plot_val')
        batch_count = 0
        for epoch in range(config.num_epoch):
            for batch_X, batch_y in stock_data.generate_one_epoch(config.batch_size):
                batch_count += 1
                sess.run(model.optimize, {
                    data: batch_X, target: batch_y, dropout: config.dropout})
            # loss train
            summary = sess.run(summary_op, {
                data: train_X, target: train_y, dropout: config.dropout})
            writer_train.add_summary(summary, epoch)
            writer_train.flush()
            # loss validation
            summary = sess.run(summary_op, {
                data: test_X, target: test_y, dropout: config.dropout})
            writer_val.add_summary(summary, epoch)
            writer_val.flush()
            # calculate train and test error
            train_cost = sess.run(model.cost, {
                data: train_X, target: train_y, dropout: config.dropout})
            train_error = sess.run(model.error, {
                data: train_X, target: train_y, dropout: config.dropout})
            test_error = sess.run(model.error, {
                data: test_X, target: test_y, dropout: config.dropout})
            print('Epoch {:2d} cost: {:6.3f} train error: {:4.2f}% test error: {:4.2f}%'.format(epoch + 1, 
                                                                              100 * train_cost,
                                                                              100 * train_error, 
                                                                              100 * test_error))
        prediction = sess.run(model.prediction, 
                        {data: test_X, target: test_y, dropout: config.dropout})
        if config.num_classes == 3:
            prediction = [pred.index(max(pred)) for pred in prediction.tolist()]
            expected = [y.index(max(y)) for y in test_y.tolist()]
            print(list(zip(prediction, expected)))
            result_count = [0, 0, 0]
            for pred in prediction:
                result_count[pred] += 1
            print(result_count)
        if config.num_classes == 1:
            prediction_percent = [pred[0] * 100 for pred in prediction.tolist()]
            expected_percent = [y[0] * 100 for y in test_y.tolist()]
            print(list(zip(prediction_percent, expected_percent)))
            prediction = [price_to_tag(pred[0], config.stay_percent) for pred in prediction.tolist()]
            expected = [price_to_tag(y[0], config.stay_percent) for y in test_y.tolist()]
            print(list(zip(prediction, expected)))
            mistakes = reduce(lambda x, item: x + 1 if item[0] != item[1] else x, 
                                list(zip(prediction, expected)), 0)
            print(mistakes / len(prediction) * 100)
            print(sess.run(model.error, {
                data: test_X, target: test_y, dropout: config.dropout}) * 100)
            print(sess.run(model.error, {
                data: test_X, target: test_y, dropout: config.dropout}) * 100)
            print(sess.run(model.error, {
                data: test_X, target: test_y, dropout: config.dropout}) * 100)

if __name__ == '__main__':
    main()