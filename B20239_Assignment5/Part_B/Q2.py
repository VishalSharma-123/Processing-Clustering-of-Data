from os import error
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse

train = pd.read_csv('abalone_train.csv')
test = pd.read_csv('abalone_test.csv')

X_train = train.iloc[:, :-1].values
Y_train = train.iloc[:, train.shape[1] - 1].values
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, test.shape[1] - 1].values

#Linear Regression model, multivariabte for training data
reggresion_train = LinearRegression().fit(X_train, Y_train)
prediction = reggresion_train.predict(X_train)
error_train = (mse(Y_train, prediction))**0.5
print ('The error in predicting the values of Rings for training data is: {}'.format(round(error_train,3)))
print ('-------------------------------------')


#Linear Regression model, multivariabte for testing data
reggresion_test = LinearRegression().fit(X_test, Y_test)
prediction_test = reggresion_test.predict(X_test)
error_test = (mse(Y_test, prediction_test))**0.5
print ('The error in predicting the values of Rings for training data is: {}'.format(round(error_test,3)))
print ('-------------------------------------')


#plot
plt.scatter(Y_test, prediction_test)
plt.xlabel('Actual Rings values')
plt.ylabel('Predicted Rings values')
plt.title('Scatter plot between Actual and Predicted Rings values')
plt.show()