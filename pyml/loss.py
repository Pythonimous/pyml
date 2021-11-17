import numpy as np


def lse(y_true, y_predict):
    """ Least squared error between two arrays of the same size """
    return 0.5 * np.sum(np.square(y_true - y_predict))


def mse(y_true, y_predict):
    """ Mean squared error between two arrays of the same size """
    return np.sum(np.square(y_true - y_predict)) / y_true.size
