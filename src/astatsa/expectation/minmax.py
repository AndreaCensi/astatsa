from contracts import contract

import numpy as np


__all__ = ['MinMax']

class MinMax():

    ''' Computes upper/lower bounds for a quantity '''

    def __init__(self):
        self.num_samples = 0
        self.value_min = None
        self.value_max = None

    @contract(value='array')
    def update(self, value):
        # TODO: check NAN, infinity
        if self.value is None:
            self.value_min = value.copy()
            self.value_max = value.copy()
        else:

            self.value_min = np.min(value, self.value_min)
            self.value_max = np.max(value, self.value_max)

        self.num_samples += 1

    def get_value(self):
        return self.value_min, self.value_max

    def get_mass(self):
        return self.num_samples
