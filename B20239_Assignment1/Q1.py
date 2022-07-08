# Name: Vishal Sharma
# Roll No: B20239


import pandas as pd                                     #importing panda module

df = pd.read_csv('pima-indians-diabetes.csv')           #accessing the file
df.drop(columns=['class'], inplace=True)                #dropping the column class, dont need it for this question

#mean of the columns
print (df.mean(), '\n')

#median of the columns
print (df.median(), '\n')

#mode of the columns
print (df.mode().iloc[0], '\n')

#maximum of the columns
print (df.max(), '\n')

#minimum of the columns
print (df.min(), '\n')

#mean of the columns
print (df.std(), '\n')
