''' Some utils for checking and comparing arrays. '''

__all__ = ['check_all_finite', 'all_finite', 'assert_allclose', 'check_allclose', 'show_some']

from contracts import contract
from numpy.ma.core import allclose
import numpy as np


try:
    from numpy.testing.utils import assert_allclose  # @UnusedImport
except ImportError:
    def assert_allclose(actual, desired, rtol=1e-7, atol=0,
                        err_msg='', verbose=True):
        ''' Backporting assert_allclose from Numpy 1.5 to 1.4 '''
        from numpy.testing.utils import assert_array_compare  # @UnresolvedImport
    
        def compare(x, y):
            return np.allclose(x, y, rtol=rtol, atol=atol)
    
        actual, desired = np.asanyarray(actual), np.asanyarray(desired)
        header = 'Not equal to tolerance rtol=%g, atol=%g' % (rtol, atol)
        assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
                             verbose=verbose, header=header)


def check_allclose(actual, desired, rtol=1e-7, atol=0,
                        err_msg='', verbose=True):
    if not allclose(actual, desired, rtol=rtol, atol=atol):
        err_msg += '\n%s != %s ' % (actual, desired)
        raise Exception(err_msg)




def assert_allequal_verbose(a, b, **kwargs):  # XXX @UnusedVariable
    failures = a != b
    if np.any(failures):
        error = show_differences(a, b, failures, "a==b")
        raise ValueError(error)


def assert_allclose_verbose(a, b, **kwargs):
    try:
        assert_allclose(a, b, **kwargs)
    except AssertionError as e:
        is_failure = a != b
        condition = 'a==b'  # XXX
        error = show_differences(a, b, is_failure, condition)
        raise Exception('%s\n\n%s' % (e, error))

#
# def close(a, b, rtol=1.e-5, atol=1.e-8):
#    """
#    Same as allclose() but returns the result element by element.
#    """
#    x = array(a, copy=False, ndmin=1)
#    y = array(b, copy=False, ndmin=1)
#    xinf = isinf(x)
#    if not all(xinf == isinf(y)):
#        return False
#    if not any(xinf):
#        return all(less_equal(absolute(x-y), atol + rtol * absolute(y)))
#    if not all(x[xinf] == y[xinf]):
#        return False
#    x = x[~xinf]
#    y = y[~xinf]
#    return all(less_equal(absolute(x-y), atol + rtol * absolute(y)))


@contract(a='shape(x)',
          b='shape(x)',
          is_failure='shape(x)',  # ,array(bool)',
          condition='str',
          MAX_N='int,>=1')
def show_differences(a, b, is_failure, condition, MAX_N=4):
    """ Returns a string summarizing the failures. """
    some, = np.nonzero(is_failure.flat)
    num = a.size
    num_fail = len(some)
    perc = 100.0 * num_fail / num
    error = ("In this array, %d/%d (%f%%) of elements do not respect "
             "the condition %s." % (num_fail, num, perc, condition))

    N = min(len(some), MAX_N)
    error += '\nThese are the first %d:' % N
    for i in range(min(len(some), MAX_N)):
        j = some[i]
        error += ('\n a.flat[%d] = %10s  b.flat[%d] = %10s ' % (j, a.flat[j],
                                                                j, b.flat[j]))
    return error


@contract(a='shape(x)',
          is_failure='shape(x)',  # ,array(bool)',
          condition='str',
          MAX_N='int,>=1')
def show_some(a, is_failure, condition, MAX_N=4):
    """ Returns a string summarizing the failures. """
    some, = np.nonzero(is_failure.flat)
    num = a.size
    num_fail = len(some)
    perc = 100.0 * num_fail / num
    error = ("In this array, %d/%d (%f%%) of elements do not respect "
             "the condition %s." % (num_fail, num, perc, condition))

    N = min(len(some), MAX_N)
    error += '\nFirst %d of %d elements:' % (N, num)
    for i in range(min(len(some), MAX_N)):
        j = some[i]
        error += ('\n a[%d] = %10s ' % (j, a.flat[j]))
    return error


def check_all_finite(value):
    """ Raises a ValueError if some values are not finite. """
    if all_finite(value):
        return

    invalid = np.logical_not(np.isfinite(value))
    msg = show_some(value, invalid, 'finite')

    msg += '\n%s %s ' % (np.min(value), np.max(value))
    raise ValueError(msg)


def all_finite(x):
    """ Fast way to check that all elements of an array are finite. """
    return np.isfinite(np.min(x)) and np.isfinite(np.max(x))

