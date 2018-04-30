class RNNConfig():
    # number of days
    num_steps = 3
    # also lstm_size
    num_features = 6001
    # number of classes
    num_classes = 2
    include_stopwords = False

    # lstm units
    num_hidden = 200
    num_layers = 3
    dropout = 0.5

    batch_size = 16
    num_epoch = 10
    # init_learning_rate = 0.001
    # learning_rate_decay = 0.99
    # init_epoch = 5
    # max_epoch = 50

    def to_dict(self):
        dct = self.__class__.__dict__
        return {k: v for k, v in dct.items() if not k.startswith('__') and not callable(v)}

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())

    
DEFAULT_CONFIG = RNNConfig()
print("Default configuration:", DEFAULT_CONFIG.to_dict())

DATA_DIR = "data"
LOG_DIR = "logs"
MODEL_DIR = "models"