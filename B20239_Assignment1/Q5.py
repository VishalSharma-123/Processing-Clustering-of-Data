# Name: Vishal Sharma
# Roll No: B20239


import pandas as pd                                     #importing panda module
from matplotlib import pyplot as plt                    #importing pyplot module
df = pd.read_csv('pima-indians-diabetes.csv')           #accessing the file

classes = df.groupby(['class'])                         #grouping by class

classes_0 = classes.get_group(0)                        #accessing class = 0
classes_1 = classes.get_group(1)                        #accessing class = 1


#for class=0, plotting histogram using hist func of pandas
classes_0.hist(column='pregs')
plt.xlabel('Number of times pregnant')
plt.title('Histogram for class = 0')
plt.show()

#for class=1
classes_1.hist(column='pregs')
plt.xlabel('Number of times pregnant')
plt.title('Histogram for class = 1')
plt.show()
