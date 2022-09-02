import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

train = pd.read_csv('abalone_train.csv')
test = pd.read_csv('abalone_test.csv')

X_train = train.iloc[:, :-1].values                                      #classifying taining data and testing data
Y_train = train.iloc[:, train.shape[1] - 1].values
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, test.shape[1] - 1].values

P = [2,3,4,5]
x = np.linspace(0, 1, 2923).reshape(-1, 1)


#part a, for training data

X = np.array(train['Shell weight']).reshape(-1, 1)
RMSE = []

for n in P:
    polynomial = PolynomialFeatures(n)                                  # n is the degree of the polynomial, creating the Polynomial object
    x_polynomial = polynomial.fit_transform(X)
    regression = LinearRegression()                                     #creating the linear regression object
    regression.fit(x_polynomial, Y_train)                               
    Y_prediction = regression.predict(x_polynomial)                     #predicting the values for training data
    error = (mse(Y_train, Y_prediction)) ** 0.5                         #computing error
    RMSE.append(error)
    print("The rmse for p=", n, 'is', round(error, 3))
    print ('-------------------------------------------')

#bar plot
plt.bar(P, RMSE)
plt.xlabel('Highest degree of polynomial')
plt.ylabel('Error')
plt.title('Unimodula Non-Linear regression for trainign data')
plt.show()

#part b, for testing data

x_test = np.array(test['Shell weight']).reshape(-1, 1)
RMSE_test = []
print ('------------------------------------------------------')

for n in P:
    polynomial = PolynomialFeatures(n)                                  # n is the degree of the polynmial, creating the Polnomial object
    x_polynomial = polynomial.fit_transform(x_test)
    regression = LinearRegression()
    regression.fit(x_polynomial, Y_test)    
    Y_prediction = regression.predict(x_polynomial)                     #predicting the value for testign data
    error = (mse(Y_test, Y_prediction)) ** 0.5                          #computing error
    RMSE_test.append(error)                                             
    print("The rmse for p=", n, 'is', round(error, 3))

#bar plot
plt.bar(P, RMSE_test)
plt.xlabel('Highest degree of polynomial')
plt.ylabel('Error')
plt.title('Unimodula Non-Linear regression for tetsing data')
plt.show()

#part c                                                                 #plot fot polynomial of best fit
Xt = np.array(train['Shell weight']).reshape(-1, 1)
xt_poly = PolynomialFeatures(5).fit_transform(Xt)                       #degree 5 is chosen at it has minimum RMSE
xp_poly = PolynomialFeatures(5).fit_transform(x)
reg = LinearRegression()
reg.fit(xt_poly, train['Rings'])
cy = reg.predict(xp_poly)
plt.scatter(train['Shell weight'], train['Rings'])
plt.plot(np.linspace(0, 1, 2923), cy, linewidth=3, color='orange')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Curve')
plt.show()

#part d

plt.scatter(Y_test, Y_prediction)
plt.xlabel('Actual Rings Values')
plt.ylabel('Predicted Rings Values')
plt.title('Scatter plot between actual and predictied values')
plt.show()