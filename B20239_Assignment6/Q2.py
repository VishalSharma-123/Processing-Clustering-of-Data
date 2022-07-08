import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

df = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],index_col=['Date'],sep=',')                    #reading the csv
train,test = train_test_split(df.values, train_size=0.65, shuffle=False)                                      #splitting the training and testing data

model = AutoReg(train, lags=5).fit()                                                                          #training AR model
coefficient = model.params                                                                                    #getting the coefficients
print (coefficient)

lagging = train[len(train)-5:]
lagging = list(lagging)
predicted = []                                                                                                # predicting values
for t in range(len(test)):
  length = len(lagging)
  lagged_values = [lagging[i] for i in range(length-5,length)] 
  y_predicted = coefficient[0]                                                                                # Initialize to w0
  for d in range(5):
    y_predicted += coefficient[d+1] * lagged_values[5-d-1]                                                    # Add other values 
  obs = test[t]
  predicted.append(y_predicted)                                                                               #Append predictions
  lagging.append(obs)                                                                                         # Append actual test value to lagged_values


plt.scatter(test, predicted)                                                                                  #scatter plot
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Scatter Plot')
plt.show()

plt.plot(test, predicted)                                                                                     #line plot
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Line Plot')
plt.show()

rmse = (math.sqrt(mean_squared_error(test, predicted))/np.mean(test))*100                                     #rmse error
print ('The RMSE error is: {}'.format(rmse))

mape = np.mean(np.abs((test - predicted)/test))*100                                                           #mape error
print('The MAPE error is: {}'.format(mape))