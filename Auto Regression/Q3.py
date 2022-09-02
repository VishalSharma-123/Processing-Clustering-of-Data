import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

df = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],index_col=['Date'],sep=',')  #reading the csv file
train,test = train_test_split(df.values, train_size=0.65, shuffle=False)                    #splitting train and test data

lag_value = [1,5,10,15,25]                                                                  #lagged values
RMSE = []
MAPE = []
for l in lag_value:
  model = AutoReg(train, lags=l).fit()
  coefficient = model.params 
  lagging = train[len(train)-l:]
  lagging = [lagging[i] for i in range(len(lagging))]
  predicted = list()                                                                        # prediction vlaues list
  for t in range(len(test)):
    length = len(lagging)
    lagged_values = [lagging[i] for i in range(length-l,length)] 
    y_predicted = coefficient[0]                                                            # Initialize to w0
    for d in range(l):
      y_predicted += coefficient[d+1] * lagged_values[l-d-1]                                # Add other values 
    obs = test[t]
    predicted.append(y_predicted)                                                           #Append predictions
    lagging.append(obs)                                                                     # Append actual test value to

  # computing rmse
  rmse_per = (math.sqrt(mean_squared_error(test, predicted))/np.mean(test))*100
  RMSE.append(rmse_per)

  # computing MAPE
  mape = np.mean(np.abs((test - predicted)/test))*100
  MAPE.append(mape)

data = {'Lag value':lag_value,'RMSE(%)':RMSE, 'MAPE' :MAPE}
print('Table 1\n',pd.DataFrame(data))

# plotting RMSE(%) vs. time lag
plt.xlabel('Lag')
plt.ylabel('RMSE(%)')
plt.title('RMSE(%) vs. time lag')
plt.xticks([1,2,3,4,5],lag_value)
plt.bar([1,2,3,4,5],RMSE)
plt.show()

# plotting MAPE vs. time lag
plt.xlabel('Lag')
plt.ylabel('MAPE')
plt.title('MAPE vs. time lag')
plt.xticks([1,2,3,4,5],lag_value)
plt.bar([1,2,3,4,5],MAPE)
plt.show()