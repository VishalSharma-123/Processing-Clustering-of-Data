import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
import math

df = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],index_col=['Date'],sep=',')                      #reading the csv file
train,test = train_test_split(df.values, train_size=0.65, shuffle=False)                                        #splitting training and testing data

p = 1
while p < len(df):
  correlation = pearsonr(train[p:].ravel(), train[:len(train)-p].ravel())
  if(abs(correlation[0]) <= 2/math.sqrt(len(train[p:]))):
    print('The heuristic value for the optimal number of lags is',p-1)
    break
  p+=1

p=p-1

model = AutoReg(train, lags=p).fit()                                                                            #model training
coef = model.params                                                                                             #getting the coefficient 
lagging = train[len(train)-p:]
lagging = [lagging[i] for i in range(len(lagging))]
predicted = list()                                                                                              # predicted values list
for t in range(len(test)):
  length = len(lagging)
  lagged_values = [lagging[i] for i in range(length-p,length)] 
  y_predicted = coef[0]                                                                                         # Initialize to w0 coefficient 
  for d in range(p):
    y_predicted += coef[d+1] * lagged_values[p-d-1]                                                             # Add other values  in y_predicted
  obs = test[t]
  predicted.append(y_predicted)                                                                                 #Append predictions 
  lagging.append(obs)                                                                                           # Append actual test value to lagged_value, to be used in next step.


rmse = (math.sqrt(mean_squared_error(test, predicted))/np.mean(test))*100                                       #Computing RMSE error
print ('The RMSE error is: {}'.format(rmse))

mape = np.mean(np.abs((test - predicted)/test))*100                                                             #Computing MAPE error
print('The MAPE error is: {}'.format(mape))