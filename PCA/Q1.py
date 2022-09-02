from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('pima-indians-diabetes.csv')
df.drop(columns = ['class'], inplace=True)
copy = df.copy()

for i in df.columns:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*(iqr)
    upper = q3+ 1.5*(iqr)
    median = df[i].median()

    for j in range(len(df[i])):
        if (df[i][j] < lower or df[i][j] > upper):
            df.at[j,i] = median
max_old = df.max()
min_old = df.min()


for i in df.columns:
    min = df[i].min()
    max = df[i].max()

    df[i] = ((df[i] - min)*7)/(max - min) + 5


print ('For dataset before normalisation:')
print ('Maximum value:\n{}\n'.format(max_old))
print ('Minimum value:\n{}\n'.format(min_old))

print ('For dataset after normalisation:')
print ('Maximum value:\n{}\n'.format(df.max()))
print ('Minimum value:\n{}\n'.format(df.min()))

# Q1 part b
mean = copy.mean()
std = copy.std()

print ('For dataset before standardisation:')
print ('Mean :\n{}\n'.format(mean))
print ('Standard Deviation :\n{}\n'.format(std))

for i in copy.columns:
    copy[i] = (copy[i] - mean[i])/(std[i])

print ('For dataset after standardisation:')
print ('Maximum value:\n{}\n'.format(copy.mean()))
print ('Minimum value:\n{}\n'.format(copy.std()))