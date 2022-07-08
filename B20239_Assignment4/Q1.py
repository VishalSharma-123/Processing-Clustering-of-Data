import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('SteelPlateFaults-2class.csv')                                                                                                 #reading the csv file
a = df.groupby('Class')                                                                                                                         #grouping by the attribute class

class0_train , class0_test = train_test_split(a.get_group(0), test_size=0.3, random_state=42, shuffle=True)                                     #separating the data into 7:3 for class 0
class1_train , class1_test = train_test_split(a.get_group(1), train_size=round(len(a.get_group(0))*0.7), random_state=42, shuffle=True)         #separating the data into 7:3 for calss 1

train = pd.concat([class0_train, class1_train])                                                                                                 #combing the trainign data
test = pd.concat([class0_test, class1_test])                                                                                                    #combining the test data

train.to_csv('SteelPlateFaults-train.csv', encoding = 'utf-8')                                                                                  #addidng trainign data to a new csv 
test.to_csv('SteelPlateFaults-test.csv', encoding = 'utf-8')                                                                                    #adding test data to a new csv

X_train = train.iloc[:, :-1]                                                                                                                    #separating the class label for test and training dtata
X_test = test.iloc[:, :-1]

X_label_train = train.iloc[:, 27]
X_label_test = test.iloc[:, 27]

for k in [1,3,5]:           
    classifier = KNeighborsClassifier(n_neighbors=k)                                                                                            #construction KNN classifier object
    classifier.fit(X_train, X_label_train)                                                                  

    X_label_predict = classifier.predict(X_test)                                                                                                #predicting th values for test data

    print ('For k = {}\n'.format(k))
    print ('The confusion matrix is:')
    print (confusion_matrix(X_label_test, X_label_predict))                                                                                     #printing the confusion matrix
    print ('\nThe accuracy for this is: {}'.format(accuracy_score(X_label_test, X_label_predict)))                                              #printing hhr accuracy
    print ('---------------------------------------------------------------------------')
