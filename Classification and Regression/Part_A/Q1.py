import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

train = pd.read_csv("SteelPlateFaults-train.csv")                                                              #reading the csv files of training and testing data
test = pd.read_csv("SteelPlateFaults-test.csv")
test_label = test['Class'].values                                                                              #reading the class label for testing data

train = train.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)                  #dropping the following attributes
test = test.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)

train_class0 = train.groupby('Class').get_group(0).to_numpy()                                                   #seperating the given data for class 0
train_class1 = train.groupby('Class').get_group(1).to_numpy()                                                   #seperating the given data for class 1
train_class1 = np.delete(train_class1, 23, axis=1)                                                              #dropping the class column from training data
train_class0 = np.delete(train_class0, 23, axis=1)

train = train.drop(['Class'], axis=1)                                                                           #dropping the class attribute from training and testing data
test = test.drop(['Class'], axis=1)

Q = [2, 4, 8, 16]                                                                                               #value of n_components
for q in Q:
    prediction = []
    gmm_class0 = GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-5).fit(train_class0)      #GMM object and fitting the traiing data 
    gmm_class1 = GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-5).fit(train_class1)
    # computing the weighted log probabilities
    log_class0 = gmm_class0.score_samples(test) + np.log(len(train_class0) / len(train))                        #calculating log values for class 0 and 1
    log_class1 = gmm_class1.score_samples(test) + np.log(len(train_class1) / len(train))
    for i in range(len(log_class0)):                                                                            #predicting the class
        if log_class0[i] > log_class1[i]:
            prediction.append(0)

        else:
            prediction.append(1)
    confusion = confusion_matrix(test_label, prediction)                                                        #confusion matrix
    accuracy = round(accuracy_score(test_label, prediction), 3)                                                 #accuracy 
    print("The confusion matrix for Q = {} is:\n{}".format(q,confusion))                                        
    print("The classification accuracy for Q = {} is {}".format(q, accuracy))
    print ("---------------------------------------------------------------------------------")

