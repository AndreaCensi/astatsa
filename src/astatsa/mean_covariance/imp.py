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

from numpy.linalg.linalg import pinv, LinAlgError

from astatsa.expectation import Expectation
import numpy as np

from ..utils import outer
from .cov2corr_imp import cov2corr


__all__ = ['MeanCovariance']


class EstimateAssocOp():
    
    def __init__(self, operation):
        self.num_samples = 0
        self._value = None
        self.operation = operation

    def update(self, value):
        if self._value is None:
            self._value = value.copy()
        else:
            self._value = self.operation(self._value, value)
        self.num_samples += 1
        
    def get_value(self):
        if self.get_num_samples() == 0:
            msg = 'No samples seen'
            raise ValueError(msg)
        return self._value
    
    def get_num_samples(self):  
        return self.num_samples
    
    def merge(self, other):
        if self.get_num_samples() == 0:
            if other.get_num_samples() > 0:
                self.num_samples = other.get_num_samples()
                self._value = other.get_value().copy()
        else:
            self.num_samples += other.get_num_samples()
            self._value = self.operation(self._value,other.get_value())
            
def np_maximum(a, b):
    return np.maximum(a, b)

def np_minimum(a, b):
    return np.minimum(a, b)


class EstimateMax(EstimateAssocOp):
    def __init__(self):
        EstimateAssocOp.__init__(self, np_maximum)
    

class EstimateMin(EstimateAssocOp):
    def __init__(self):
        EstimateAssocOp.__init__(self, np_minimum)



class MeanCovariance(object):
    ''' Computes mean and covariance of a quantity '''

    def __init__(self, max_window=None):
        self.mean_accum = Expectation(max_window)
        self.covariance_accum = Expectation(max_window)
        self.est_min = EstimateMin()
        self.est_max = EstimateMax()
        self.num_samples = 0
        self.shape = None
        
    def merge(self, other):
        assert isinstance(other, MeanCovariance)
        self.mean_accum.merge(other.mean_accum)
        self.covariance_accum.merge(other.covariance_accum)
        self.est_min.merge(other.est_min)
        self.est_max.merge(other.est_max)
        self.num_samples += other.num_samples
        warnings.warn('minimum/maximum missing')


    def get_num_samples(self):
        return self.num_samples

    def update(self, value, dt=1.0):
        if self.num_samples == 0:
            self.shape = value.shape
        else:
            if not (value.shape == self.shape):
                msg = ('Value shape changed: %s -> %s' %
                        (self.shape, value.shape))
                raise ValueError(msg)

        self.est_min.update(value)
        self.est_max.update(value)

        self.mean_accum.update(value, dt)
        mean = self.mean_accum.get_value()
        value_norm = value - mean

        P = outer(value_norm, value_norm)
        self.covariance_accum.update(P, dt)
        self.last_value = value

        self.num_samples += dt

    def assert_some_data(self):
        if self.num_samples == 0:
            raise Exception('Never updated')

    def get_mean(self):
        self.assert_some_data()
        return self.mean_accum.get_value()

    def get_maximum(self):
        self.assert_some_data()
        return self.est_max.get_value()

    def get_minimum(self):
        self.assert_some_data()
        return self.est_min.get_value()

    def get_covariance(self):
        self.assert_some_data()
        return self.covariance_accum.get_value()

    def get_correlation(self):
        self.assert_some_data()
        corr = cov2corr(self.covariance_accum.get_value())
        np.fill_diagonal(corr, 1)
        return corr

    def get_information(self, rcond=1e-2):
        self.assert_some_data()
        try:
            P = self.get_covariance()
            return pinv(P, rcond=rcond)
        except LinAlgError:
            filename = 'pinv-failure'
            import pickle
            with  open(filename + '.pickle', 'w') as f:
                pickle.dump(self, f)
            # logger.error('Did not converge; saved on %s' % filename)

    

        
