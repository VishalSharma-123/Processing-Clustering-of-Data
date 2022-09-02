import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
import statsmodels.api as sm



df = pd.read_csv('daily_covid_cases.csv')                                                                               #reading the csv file

x = [16]
for i in range(10):
    x.append(x[i] + 60)

labels = ['Feb-20', 'Apr-20', 'Jun-20', 'Aug-20', 'Oct-20', 'Dec-20', 'Feb-21', 'Apr-21', 'Jun-21', 'Aug-21', 'Oct-21']
original = df['new_cases']
plt.figure(figsize=(20, 10))                                                                                            #plot for the cases
plt.xticks(x, labels)
plt.xlabel('Month-Year')
plt.ylabel('New Confirmed Cases')
plt.title('Cases vs Months')
plt.plot(original)
plt.show()

shifted = df.copy()
shifted_data = shifted['new_cases'].shift(1)                                                                            #shifting the data by a day
original_data = df['new_cases']

corr = pearsonr(shifted_data[1:], original_data[1:])                                                                    #correlation between lagged and original data
print ('auto correlation {}'.format(round(corr[0],3)))

plt.scatter(shifted_data[1:], original_data[1:])                                                                        #scatter plot
plt.title('Scatter Plot')
plt.xlabel('Shifted data')
plt.xlabel('original data')
plt.show()

correlation = []
p = [1,2,3,4,5,6]                                                                                                       #comuting correlation for different lagged values

for i in p:
    shifted_data = shifted['new_cases'].shift(i)
    corr = pearsonr(shifted_data[i:], original_data[i:])
    correlation.append(corr[0])


plt.plot(p,correlation)
plt.xlabel('Value of p')
plt.ylabel('Value of correlation')
plt.show()

sm.graphics.tsa.plot_acf(original,lags=p)
plt.show()