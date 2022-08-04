from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


def soft_th(x, lambda_):
    return np.sign(x) * np.maximum(abs(x) - lambda_, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.w_ = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X_ = np.c_[np.ones(X.shape[0]), X]
        else:
            X_ = X

        self.w_ = np.linalg.solve(X_.T @ X_, X_.T @ y)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            X_ = np.c_[np.ones(X.shape[0]), X]
        else:
            X_ = X

        return X_ @ self.w_


class Ridge(LinearRegression):
    def __init__(self, fit_intercept=True, lambda_=1.):
        super().__init__(fit_intercept)
        self.lambda_ = lambda_

    def fit(self, X, y):
        if self.fit_intercept:
            X_ = np.c_[np.ones(X.shape[0]), X]
        else:
            X_ = X

        I = np.eye(X_.shape[1])
        self.w_ = np.linalg.solve(X_.T @ X_ + self.lambda_ * I, X_.T @ y)


class Lasso():
    def __init__(self, lambda_=1., fit_intercept=True, tol=1e-10, max_iter=100000):
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.w_ = None

    def fit(self, X, y):
        n, d = X.shape
        w_ = np.zeros(d)
        w0 = np.zeros(1)

        for i in range(self.max_iter):
            w_old = np.copy(w_)
            if self.fit_intercept == True:
                w0 = (y - X @ w_).sum() / n
            for j in range(d):
                w_sub_j = np.copy(w_).reshape(-1)
                w_sub_j[j] = 0
                s_j = X[:, j].T @ (y - w0 - X @ w_sub_j)
                r = X[:, j] @ X[:, j]
                w_[j] = soft_th(s_j / r, self.lambda_)

            eps = np.linalg.norm(w_old - w_, 1)
            if eps < self.tol:
                break
        if self.fit_intercept == True:
            self.w_ = np.append(w0, w_)
        else:
            self.w_ = w_

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X_ = np.c_[np.ones(X.shape[0]), X]
        else:
            X_ = X

        return X_ @ self.w_


class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X):
        X_ = []
        X_1 = X.reshape(1, -1)
        for deg in range(self.degree + 1):
            X_.append(X_1**deg)

        return np.concatenate(X_).T


class PolynomialRegression:
    def __init__(self, regressor=LinearRegression(fit_intercept=False), degree=1):
        self.regressor = regressor
        self.degree = degree

    def fit(self, X, y):
        polynomial_features = PolynomialFeatures(self.degree)
        X_poly = polynomial_features.fit_transform(X)
        self.regressor.fit(X_poly, y)

    def predict(self, X):
        polynomial_features = PolynomialFeatures(self.degree)
        X_poly = polynomial_features.fit_transform(X)
        return self.regressor.predict(X_poly)


THRESHMIN = 1e-10


class LogisticRegression:
    def __init__(self, tol=0.001, max_iter=3, fit_intercept=True, random_seed=0):
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.random_state = np.random.RandomState(random_seed)
        self.w_ = None

    def fit(self, X, y):
        if self.fit_intercept:
            X_ = np.c_[np.ones(X.shape[0]), X]
        else:
            X_ = X

        self.w_ = self.random_state.randn(X_.shape[1])
        diff = np.inf
        w_prev = self.w_
        iter = 0
        while diff > self.tol and iter < self.max_iter:
            yhat = sigmoid(np.dot(X_, self.w_))
            r = np.clip(yhat * (1 - yhat), THRESHMIN, np.inf)
            XR = X_.T * r
            XRX = np.dot(X_.T * r, X_)
            w_prev = self.w_
            b = np.dot(XR, np.dot(X_, self.w_) - 1 / r * (yhat - y))
            self.w_ = np.linalg.solve(XRX, b)
            diff = abs(w_prev - self.w_).mean()
            iter += 1

    def predict(self, X):
        if self.fit_intercept:
            X_ = np.c_[np.ones(X.shape[0]), X]
        else:
            X_ = X
        yhat = sigmoid(np.dot(X_, self.w_))
        return np.where(yhat > .5, 1, 0)


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
