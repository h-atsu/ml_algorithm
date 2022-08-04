from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


class LogisticRegression:
    def __init__(self, tol=0.001, max_iter=3, fit_intercept=True, random_seed=0):
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.random_state = np.random.RandomState(random_seed)
        self.w_ = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        if self.fit_intercept:
            X_ = np.c_[np.ones(X.shape[0]), X]
        else:
            X_ = X


##### main ######################
data = datasets.load_breast_cancer()

if __name__ == '__main__':
    X = data['data']
    y = data['target']

    lr = LogisticRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    (y_pred == y).sum()
    print(y)
