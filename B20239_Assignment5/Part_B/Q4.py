import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

train = pd.read_csv('abalone_train.csv')
test = pd.read_csv('abalone_test.csv')

X_train = train.iloc[:, :-1].values
Y_train = train.iloc[:, train.shape[1] - 1].values
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, test.shape[1] - 1].values

P = [2,3,4,5]

RMSE = []

for n in P:
    polynomial = PolynomialFeatures(n)                                              # n is the degree of polynomial
    x_polynomial = polynomial.fit_transform(X_train)
    regression = LinearRegression()
    regression.fit(x_polynomial, Y_train)
    Y_prediction = regression.predict(x_polynomial)
    error = (mse(Y_train, Y_prediction)) ** 0.5
    RMSE.append(error)
    print("The rmse for p=", n, 'is', round(error, 3))

#bar plot
plt.bar(P, RMSE)
plt.xlabel("Value of highest degree")
plt.ylabel('RMSE Error')
plt.title('Plot between degree of polynomial and error for training data')
plt.show()

#part b, for testing data

RMSE_test = []
print ('------------------------------------------------------')

for n in P:
    polynomial = PolynomialFeatures(n)                                              # n is the degree of polynomial
    x_polynomial = polynomial.fit_transform(X_test)
    regression = LinearRegression()
    regression.fit(x_polynomial, Y_test)
    Y_prediction = regression.predict(x_polynomial)
    error = (mse(Y_test, Y_prediction)) ** 0.5
    RMSE_test.append(error)
    print("The rmse for p=", n, 'is', round(error, 3))

#bar plot
plt.bar(P, RMSE_test)
plt.xlabel("Value of highest degree")
plt.ylabel('RMSE Error')
plt.title('Plot between degree of polynomial and error for testing data')
plt.show()

#part c
x_polynomial = polynomial.fit_transform(X_test)                                     #choosing p = 3 as it has minimum RMSE error
regression = LinearRegression()
regression.fit(x_polynomial, Y_test)
Y_prediction = regression.predict(x_polynomial)
plt.scatter(Y_test, Y_prediction)
plt.xlabel('Actual value of Rings')
plt.ylabel('Predicted value of Rigns')
plt.title('Multi modular Non-Linear Regression model')
plt.show()