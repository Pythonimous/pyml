import numpy as np


def mse(y_true, y_predict):
    """ Mean squared error between two arrays of the same size """
    return 0.5 * np.sum((y_true - y_predict) ** 2)
