# Name: Vishal Sharma
# Roll No: B20239


import pandas as pd                                     #importing panda module
from matplotlib import pyplot as plt                    #importing pyplot
df = pd.read_csv('pima-indians-diabetes.csv')           #accessing the file

df.drop(columns=['class'], inplace=True)                #dropping the column class, dont need it for this question


#using hist function of pandas to plot the histogram
df.hist(column='pregs')
plt.xlabel('Number of times pregnant')
plt.show()

df.hist(column='skin', color = '#00A0B0')
plt.xlabel('Triceps skin fold thickness (mm)')
plt.show()