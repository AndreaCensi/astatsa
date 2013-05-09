# Here are a couple references on computing sample variance.
#
# Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1983). 
# Algorithms for Computing the Sample Variance: Analysis and Recommendations. 
# The American Statistician 37, 242-247.
#
# Ling, Robert F. (1974). Comparison of Several Algorithms for Computing Sample 
# Means and Variances. Journal of the American Statistical Association,
# Vol. 69, No. 348, 859-866. 

import warnings
from astatsa.expectation import Expectation
from astatsa.utils.np_comparisons import check_all_finite
import numpy as np

from numpy import multiply
from cbc.tools.math_utils import cov2corr
outer = multiply.outer



class MeanCovariance:
    ''' Computes mean and covariance of a quantity '''
    
    def __init__(self, max_window=None):
        self.mean_accum = Expectation(max_window)
        self.covariance_accum = Expectation(max_window)
        self.minimum = None
        self.maximum = None  # TODO: use class
        self.num_samples = 0
        
    def merge(self, other):
        warnings.warn('To test')
        assert isinstance(other, MeanCovariance)
        self.mean_accum.merge(other.mean_accum)
        self.covariance_accum.merge(other.covariance_accum)
        self.num_samples += other.num_samples
        warnings.warn('minimum/maximum missing')


    def get_num_samples(self):
        return self.num_samples

    def update(self, value, dt=1.0):
        check_all_finite(value)

        self.num_samples += dt

        n = value.size
        if  self.maximum is None:
            self.maximum = value.copy()
            self.minimum = value.copy()
            self.P_t = np.zeros(shape=(n, n), dtype=value.dtype)
        else:
            # TODO: check dimensions
            if not (value.shape == self.maximum.shape):
                raise ValueError('Value shape changed: %s -> %s' % 
                                 (self.maximum.shape, value.shape))
            self.maximum = np.maximum(value, self.maximum)
            self.minimum = np.minimum(value, self.minimum)

        self.mean_accum.update(value, dt)
        mean = self.mean_accum.get_value()
        value_norm = value - mean

        check_all_finite(value_norm)
        P = outer(value_norm, value_norm)
        self.covariance_accum.update(P, dt)
        self.last_value = value

    def assert_some_data(self):
        if self.num_samples == 0:
            raise Exception('Never updated')

    def get_mean(self):
        self.assert_some_data()
        return self.mean_accum.get_value()

    def get_maximum(self):
        self.assert_some_data()
        return self.maximum

    def get_minimum(self):
        self.assert_some_data()
        return self.minimum

    def get_covariance(self):
        self.assert_some_data()
        return self.covariance_accum.get_value()

    def get_correlation(self):
        self.assert_some_data()
        corr = cov2corr(self.covariance_accum.get_value())
        np.fill_diagonal(corr, 1)
        return corr

