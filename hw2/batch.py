import numpy as np

class Batcher():
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self._batch_size = batch_size
        if self.x.shape[0] != self.y.shape[0]:
            raise Exception('Input data number not consistent.')
        self._total_data = self.x.shape[0]
        self._init_permutation()
        self._step_now = 0

    def new_epoch(self):
        self._permut = np.random.permutation(self._total_data)
        self._step_now = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        return self._next_batch()

    def _init_permutation(self):
        self._permut = np.random.permutation(self._total_data)

    def _next_batch(self):
        if self._batch_size * (self._step_now + 1) >= self._total_data:
            raise StopIteration()
        start = self._batch_size * self._step_now
        end = min(start + self._batch_size, self._total_data)
        ret_x = self.x[self._permut[start:end]]
        ret_y = self.y[self._permut[start:end]]
        self._step_now += 1
        return ret_x, ret_y
