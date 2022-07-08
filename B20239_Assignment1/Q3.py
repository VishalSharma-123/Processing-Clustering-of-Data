# Name: Vishal Sharma
# Roll No: B20239


import pandas as pd                         #importing panda module
df = pd.read_csv('pima-indians-diabetes.csv')           #accessing the file

df.drop(columns=['class'], inplace=True)                #dropping the column class, dont need it for this question

#correlation coefficient for every column
print (df.corr())