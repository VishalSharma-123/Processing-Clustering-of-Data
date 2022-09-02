import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('SteelPlateFaults-train.csv')                                                  #reading the training and test data
df1 = pd.read_csv('SteelPlateFaults-test.csv')

class1 = df['Class']                                                                            #storing class label of training and testing data
class2 = df1['Class']

df.drop(df.columns[0], axis=1, inplace=True)
df1.drop(df1.columns[0], axis=1, inplace=True)


df.drop(columns=['Class'], inplace = True)                                                      #dropping the class attribute
df1.drop(columns=['Class'], inplace = True)

normalized_train=(df-df.min())/(df.max()-df.min())                                              #Normalizing the training data
normalized_test = (df1-df.min())/(df.max() - df.min())                                          #normalizing testing data

normalized_train['Class'] = class1                                                              #adding the attribute class to training and testing data
normalized_test['Class'] = class2

X_train = normalized_train.iloc[:, :-1]                                                         #reading training and testing data
X_test = normalized_test.iloc[:, :-1]

X_label_train = normalized_train.iloc[:, 27]                                                    #reading the class label for training and testing data
X_label_test = normalized_test.iloc[:, 27]

for k in [1,3,5]:                                                                               #the knn classifier
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, X_label_train)

    X_label_predict = classifier.predict(X_test)

    print ('For k = {}\n'.format(k))
    print ('The confusion matrix is:')
    print (confusion_matrix(X_label_test, X_label_predict))
    print ('\nThe accuracy for this is: {}'.format(accuracy_score(X_label_test, X_label_predict)))
    print ('---------------------------------------------------------------------------')