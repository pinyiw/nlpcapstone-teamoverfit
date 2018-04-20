from time import time


class Timer:
    def __init__(self):
        self.start_time = time()

    def __enter__(self):
        self.cur_start_time = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cur_end_time = time()
        print('Time taken: %fs' % (self.cur_end_time - self.cur_start_time))