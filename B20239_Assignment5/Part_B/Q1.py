from ast import Index
from textwrap import indent
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv('abalone.csv')                                                         #reading the data and classifying training and testing data
train, test = train_test_split(df, train_size=0.7, random_state=42, shuffle=True)
train.to_csv('abalone_train.csv', index = False)
test.to_csv('abalone_test.csv', index = False)

#Finding attribute with maximum correlation with Rings
corr = df.corr()['Rings']
corr1 = list(corr.copy())
corr.pop('Rings')
ind = corr1.index(corr.max())
attribute = corr.index[ind]

#Linear Regression Model
x = train[attribute]
y = train['Rings']

#finding the slope and constant for univariante regression model
m,c = np.polyfit(x,y,1)
plt.plot(x,y,'x')
plt.plot(x, m*x + c, linewidth = 2.5)
plt.xlabel(attribute)
plt.ylabel('Predicted Rings values')
plt.title('Plot for Predicted Rings values from {}'.format(attribute))
plt.show()

#predictiing training data
y_train_predict = m*x + c
error = (mse(y, y_train_predict))**0.5
print ('The error in predicting the values of Rings for training data is: {}'.format(round(error,3)))
# print ('The accuracy in predicting the values of Rings for testing data is: {}'.format(accuracy))
print ('-------------------------------------')

#predicting test data
y_test_predict = m*test[attribute] + c
error = (mse(test['Rings'], y_test_predict))**0.5
print ('The error in predicting the values of Rings for testing data is: {}'.format(round(error,3)))
# print ('The accuracy in predicting the values of Rings for testing data is: {}'.format(accuracy))
print ('-------------------------------------')


#plot 
plt.scatter(test['Rings'], y_test_predict, marker='*')
plt.xlabel('Actual Rings values')
plt.ylabel('Predicted Rings values')
plt.title('Scatter plot between Actual and Predicted Rings values')
plt.show()