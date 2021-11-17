from .context import pyml

import unittest
import numpy as np


class LossTestSuite(unittest.TestCase):
    """ Tests for different loss functions """

    x01 = np.array([1, 1, 2])
    x02 = np.array([1, 1, 0])
    x03 = np.array([0, 0, 0])
    x04 = np.array([1, 1, 1])
    x05 = np.array([1, 2, 3, 4, 5, 6])
    x06 = np.array([7, 8, 9, 10, 11, 12])
    x07 = np.array([100, 123, 532])
    x08 = np.array([60, 127, 675])
    x09 = np.array([0.51, 0.2, 0.95, 0.438, 0.616])
    x10 = np.array([0, 0, 1, 0, 1])

    def test_lse(self):
        self.assertEqual(pyml.loss.lse(self.x01, self.x02), 2)
        self.assertEqual(pyml.loss.lse(self.x03, self.x04), 1.5)
        self.assertEqual(pyml.loss.lse(self.x05, self.x06), 108)
        self.assertEqual(pyml.loss.lse(self.x07, self.x08), 11032.5)
        self.assertEqual(pyml.loss.lse(self.x09, self.x10), 0.32095)

    def test_mse(self):
        self.assertEqual(pyml.loss.mse(self.x01, self.x02), 4/3)
        self.assertEqual(pyml.loss.mse(self.x03, self.x04), 1)
        self.assertEqual(pyml.loss.mse(self.x05, self.x06), 36)
        self.assertEqual(pyml.loss.mse(self.x07, self.x08), 7355)
        self.assertEqual(pyml.loss.mse(self.x09, self.x10), 0.12838)
