import numpy as np

class LinearRegression(object):

    def __init__(self):
        self.coef = None

    def step(self, X, y, start, end, alpha):
        self.coef -= alpha * (y[start:end, :] - self.coef.T * (X[start:end, :]))*X[start:end, :]

    def gradient_descent(self, X, y, batch_size=-1, alpha=0.01):

        num_examples = X.shape[0]

        if batch_size == -1:
            batch_size = num_examples

        batch_start = 0
        batch_end = 0

        self.coef = np.zeros((1, X.shape[1]))

        while True:

            while batch_end < num_examples:
                prev_coef = self.coef
                self.step(X, y, batch_start, batch_end, alpha)
                if prev_coef == self.coef: return
                batch_start = batch_end
                batch_end += batch_size

            batch_end = num_examples
            prev_coef = self.coef
            self.step(X, y, batch_start, batch_end, alpha)
            if prev_coef == self.coef: return

    def normal_equations(self, X, y):
        inverted = np.linalg.inv(np.matmul( X.T, X ))
        self.coef = np.matmul( np.matmul( inverted, X.T ), y)

    def fit(self, X, y, method='normal'):
        methods = {'lms': self.gradient_descent,
                   'normal': self.normal_equations}
        X = np.insert(X, 0, 1, axis=1)
        methods.get(method, self.normal_equations)(X, y)

    def infer(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.matmul(X, self.coef)

    @staticmethod
    def mse(y_true, y_predict):
        return 0.5 * np.sum((y_true - y_predict)**2)

    def evaluate(self, X, y_true):
        y_predict = self.infer(X)
        return self.mse(y_true, y_predict)
