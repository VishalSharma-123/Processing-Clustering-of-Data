from pandas.core.frame import DataFrame
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('pima-indians-diabetes.csv')
df.drop(columns = ['class'], inplace=True)

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

copy = df.copy()
mean = copy.mean()
std = copy.std()

for i in copy.columns:
    copy[i] = (copy[i] - mean[i])/(std[i])

covariance = copy.cov()

pca = PCA(n_components=2)
pca.fit(copy)
matrix = pca.transform(copy)
df_new = pd.DataFrame(data = matrix, columns = ['x1', 'x2'])
a = df_new.cov().round(3)
plt.scatter(df_new['x1'], df_new['x2'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot for reduced Data')
plt.show()

eigen, eigen_vector = np.linalg.eig(covariance)
print ('The variance of column "x1" and "x2" are: {} {}'.format(a['x1']['x1'], a['x2']['x2']))
print ('The eige values of these columns are: {} {}'.format(round(eigen[0],3), round(eigen[1],3)))

#part b
eigen.sort()
decreasing = eigen[::-1]
plt.plot(decreasing, color = 'black')
plt.ylabel('Eigen Value')
plt.title('The eigen values in decreasing order')
plt.show()


#part c

error_list = []
l = []

for i in range(1,9):
    l.append(i)
    pca = PCA(n_components=i)
    matrix=pca.fit_transform(copy)                                                      #Data with Reduced Dimension
    inverse=pca.inverse_transform(matrix)                                               #Reconstructed Data
    error_list.append((((copy.values-inverse)**2).mean())**.5)                          #Appending list with RMSE

    if (i>1):
        matrix = pd.DataFrame(data = matrix)
        print ('For l = {}, the covariance matrix is: \n'.format(i))
        print (matrix.cov().round(3))


plt.bar(l,error_list)
plt.plot(l,error_list, color = 'black')
plt.scatter(l,error_list, color = 'red')
plt.xlabel('Value of l')
plt.ylabel('Mean Error')
plt.title('Plot of Error vs l')
plt.show()

#part d
pca = PCA(n_components=8)
matrix = pca.fit_transform(copy)
matrix = pd.DataFrame(data = matrix)
print ('For the original matrix, covariance matrix is:\n')
print (copy.cov().round(3), '\n')

print ('For the matrix after PCA transformation:\n')
print (matrix.cov().round(3))
