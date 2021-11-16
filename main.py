import pandas as pd
import numpy as np

from classical.linear_model import LinearRegression

seed = 42

dataset = pd.read_csv('data/external/housing.csv').sample(frac=1, random_state=seed).reset_index(drop=True)

# print(data.columns)


def test_linear(train_dataset):

    data = train_dataset.dropna()
    X = np.array(data[['total_rooms', 'total_bedrooms', 'median_income']])
    y = np.array(data['median_house_value']).reshape((X.shape[0],1))
    X_train, X_test = X[:20000, :], X[20000:, :]
    y_train, y_test = y[:20000, :], y[20000:, :]

    linear_regression = LinearRegression()

    linear_regression.fit(X_train, y_train)

    predictions = linear_regression.infer(X_test)

    print(linear_regression.evaluate(X_test, y_test))

    return predictions


test_linear(dataset)

