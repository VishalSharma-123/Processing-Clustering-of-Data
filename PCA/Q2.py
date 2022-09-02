import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

mean = np.array([0,0])
cov = np.array([[13, -3], [-3, 5]])
data = np.random.multivariate_normal(mean = mean, cov=cov, size =1000, check_valid = 'ignore')
df = pd.DataFrame(data = data, columns = ['Column1', 'Column2'])

plt.scatter(df['Column1'], df['Column2'], marker = '*')
plt.title('Scatter plot for 2-D data sample')
plt.xlabel('Column1')
plt.ylabel('Column2')
plt.show()

#part b
value, vector = np.linalg.eig(cov)
eig_vec1 = vector[:,0]
eig_vec2 = vector[:,1]
origin = [0,0]
print ('The eigen values are: {} and {}'.format(value[0], value[1]))
plt.scatter(df['Column1'], df['Column2'], marker = '*')
plt.quiver(*origin, *eig_vec1, color=['r'], scale=5)
plt.quiver(*origin, *eig_vec2, color=['b'], scale=4)
plt.title('Plot for 2D data with the eigen vectors')
plt.xlabel('Column1')
plt.ylabel('Column2')
plt.show()

#paet c
projection = np.dot(df,vector)

#For 1st eigen vector

plt.scatter(df['Column1'], df['Column2'], marker = '*')
plt.quiver(*origin, *eig_vec1, color=['r'], scale=5)
plt.quiver(*origin, *eig_vec2, color=['b'], scale=4)
plt.scatter(projection[:,0]*eig_vec1[0],projection[:,0]*eig_vec1[1],color='magenta',marker='*')
plt.title('Plot for 2D data with the eigen vectors with projection on first vector')
plt.xlabel('Column1')
plt.ylabel('Column2')
plt.show()

#for 2nd eigen vector

plt.scatter(df['Column1'], df['Column2'], marker = '*')
plt.quiver(*origin, *eig_vec1, color=['r'], scale=5)
plt.quiver(*origin, *eig_vec2, color=['b'], scale=4)
plt.scatter(projection[:,0]*eig_vec2[0],projection[:,0]*eig_vec2[1],color='magenta',marker='*')
plt.title('Plot for 2D data with the eigen vectors with projection on second vector')
plt.xlabel('Column1')
plt.ylabel('Column2')
plt.show()

#part d
inverse = np.dot(projection, np.transpose(vector))
error = ((data - inverse)**2).mean()
print ('The mean recosnstruction error is: {}'.format(error))