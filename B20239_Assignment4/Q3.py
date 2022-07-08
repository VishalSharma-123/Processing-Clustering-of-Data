from os import truncate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import multivariate_normal


df = pd.read_csv('SteelPlateFaults-train.csv')                                                                  #reading testing and training data
df1 = pd.read_csv('SteelPlateFaults-test.csv')

df.drop(columns = ['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', df.columns[0]], inplace=True)                  #deleting these columns as correlation is 1
df1.drop(columns = ['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', df1.columns[0]], inplace=True)

train_class0 = df[df['Class'] == 0]                                                                             #classifiyin training data based on class  
train_class1 = df[df['Class'] == 1]

print ("Mean of data for class 0\n")                                                                            #printing the mean of these class separately
print (train_class0.mean().round(3))
print ('\nMean of data for class 1\n')
print (train_class1.mean().round(3))
print ('------------------------------------------------------')

a = round(train_class0.cov())                                                                                   #adding the covariance of these attributes to a new csv file
b = round(train_class0.cov())

a.to_csv('Covariance_class0.csv')
b.to_csv('Covariance_class1.csv')

train_label_class0 = train_class0.pop('Class')                                                                  #assigning the class attribute to a new variable
train_label_class1 = train_class1.pop('Class')

mean_class0 = np.array(train_class0.mean())                                                                     #constructuing mean array for claass 0 and class 1
covariance_class0 = np.array(train_class0.cov())                                                                #cosntructing covariacne array for class 0 and class 1
classs0_distr=multivariate_normal(mean_class0,covariance_class0,allow_singular=True)                            #cosntructing multivariaate normal distribution object for class 0 and class 1

mean_class1 = np.array(train_class1.mean())                                                                     
covariance_class1 = np.array(train_class1.cov())
classs1_distr=multivariate_normal(mean_class1,covariance_class1,allow_singular=True)

test_label_data = df1.pop('Class')
test_data = df1

#Prior probability  
prior_class0 = len(train_class0)/len(df)    
prior_class1 = len(train_class1)/len(df)


prediction = []

for idx,rows in test_data.iterrows():
    probability_class0 = classs0_distr.pdf(list(rows))*prior_class0                                             #calculating the probability for both class 0 and class 1
    probability_class1 = classs1_distr.pdf(list(rows))*prior_class1

    if (probability_class0 > probability_class1):
        prediction.append(0)
    else:
        prediction.append(1)

print ('The confusion matrix is:')
print (confusion_matrix(test_label_data,prediction))                                                            #printing the confusion matrix and accuracy
print('\nThe accuracy for this is: {}'.format(accuracy_score(test_label_data, prediction)))