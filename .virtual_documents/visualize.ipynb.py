import numpy as np 
import matplotlib.pyplot as plt
from linear import LinearRegression,Ridge,PolynomialFeatures,PolynomialRegression, Lasso, Lasso2
from sklearn.linear_model import Lasso as sk_Lasso
np.random.seed(10)


X=np.random.random((10,1))
y = 3*X + np.random.random(X.shape) + 3


plt.scatter(X,y)


lr = LinearRegression(fit_intercept=False)
lr.fit(X,y)
y_pred = lr.predict(X)


x = np.linspace(X.min()-0.1,X.max()+0.1).reshape(-1,1)


x.shape


plt.scatter(X, y, label = "true data")
plt.plot(x, lr.predict(x), label = "prediction", c='r')
plt.legend()


ri = Ridge()
ri.fit(X,y)
y_pred = ri.predict(X)


plt.scatter(X, y, label = "true data")
plt.plot(x, ri.predict(x), label = "prediction", c='r')
plt.legend()


lr = sk_Lasso()
lr.fit(X,y)
y_pred = lr.predict(X)


y_pred


lr.coef_


plt.scatter(X, y, label = "true data")
plt.plot(x, lr.predict(x), label = "prediction", c='r')
plt.legend()


# ri = Lasso()
# ri.fit(X,y)
# y_pred = ri.predict(X)


# plt.scatter(X, y, label = "true data")
# plt.plot(x, ri.predict(x), label = "prediction", c='r')
# plt.legend()


X = np.random.random(30)
eps = np.random.uniform(len(X))
y = np.sin(X*10 + eps) + np.exp(-X)
pl = PolynomialRegression(Ridge(lambda_=1e-10, fit_intercept=False) ,degree=10)
pl.fit(X,y)
y_pred = pl.predict(X)
plt.scatter(X, y, label = "true data")
x = np.linspace(X.min()-0.1, X.max()+0.1)
#x = np.linspace(X.min()-0.3, X.max()+0.3)
plt.plot(x, pl.predict(x), label = "prediction", c='r')
plt.legend()


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
data = datasets.load_wine()


X = data['data']
y = data['target']


X


y


X_train, X_test, y_train, y_test = train_test_split(X,y)


lr = LinearRegression()


lr.fit(X_train,y_train)


y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)


train_mse = mean_squared_error(y_pred_train, y_train)
test_mse = mean_squared_error(y_pred_test, y_test)
print("train MSE : {:.4}".format(train_mse))
print("test MSE : {:.4}".format(test_mse))


for lambda_ in [1., 0.1, 0.01]:
    model = Lasso(lambda_=lambda_)
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_mse = mean_squared_error(y_pred_train, y_train)
    test_mse = mean_squared_error(y_pred_test, y_test)
    print('-'*3, "lambda = {}".format(lambda_), '-'*3)
    print("train MSE : {:.4}".format(train_mse))
    print("test MSE : {:.4}".format(test_mse))
    print('coef')
    print(model.w_)        


def f(x):
    return 1/(1+x)


def sample(n):
    x = np.random.random(n) * 5
    y = f(x)
    return x, y 


xs = np.linspace(0,5,500)
n=10000

y_pr_sum = np.zeros(len(xs))
y_lr_sum = np.zeros(len(xs))

for _ in range(n):
    x, y = sample(5)
    pr = PolynomialRegression(degree=4)
    lr = LinearRegression()
    pr.fit(x,y)
    lr.fit(x,y)
    y_pr = pr.predict(xs)
    y_lr = lr.predict(xs)
    
    y_pr_sum += y_pr
    y_lr_sum += y_lr       
    
E_y_pr = y_pr_sum/n
E_y_lr = y_lr_sum/n


plt.plot(xs, f(xs), label = 'truth')
plt.plot(xs, E_y_pr, linestyle = "dashed" ,label = 'polynomial reg')
plt.plot(xs, E_y_lr, linestyle = "dotted", label = 'linear reg')
plt.legend()


y_true = f(xs)

# 合計値
y_pr_sum = np.zeros(len(xs))
y_lr_sum = np.zeros(len(xs))

# f(x)との二乗誤差
y_pr_se = np.zeros(len(xs))
y_lr_se = np.zeros(len(xs))

for _ in range(n):
    x, y = sample(5)    
    pr = PolynomialRegression(degree=4)
    lr = LinearRegression()
    pr.fit(x,y)
    lr.fit(x,y)
    y_pr = pr.predict(xs)
    y_lr = lr.predict(xs)
    
    y_pr_sum += y_pr
    y_lr_sum += y_lr
    y_pr_se += (y_true - y_pr)**2
    y_lr_se += (y_true - y_lr)**2
    
bi_y_pr = (y_true - y_pr_sum/n)**2
bi_y_lr = (y_true - y_lr_sum/n)**2
var_y_pr = y_pr_se / n
var_y_lr = y_lr_se / n


plt.plot(xs, y_pr_se)


plt.plot(xs, bi_y_pr)


fig = plt.figure(figsize = (10,5))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title("linear")
ax2.set_title("polynomial")
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)

ax1.fill_between(xs, 0, bi_y_lr, color="0.2",label = "bias")
ax1.fill_between(xs, bi_y_lr, var_y_lr, color = "0.7",label = "variance")
ax1.legend()
ax2.fill_between(xs, 0, bi_y_pr, color="0.2",label = "bias")
ax2.fill_between(xs, bi_y_pr, var_y_pr, color="0.7",label = "variance")
ax2.legend()


a = [[0,0,0],[0,0,0]]


a


b[:][:] = a[:][:][:]


b[0][0] = 200


b


a
