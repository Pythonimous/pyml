from .context import pyml

import unittest
import numpy as np


class MetricsTestSuite(unittest.TestCase):
    """ Tests for metrics """

    def test_mse(self):
        x01, x02 = np.array([1, 1, 2]), np.array([1, 1, 0])
        self.assertEqual(pyml.metrics.mse(x01, x02), 2)
