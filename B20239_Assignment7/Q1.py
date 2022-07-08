import pandas as pd                                                                             #importing all the modules
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

df = pd.read_csv('Iris.csv')                                                                    #reading the csv file
test = df['Species']
class_label = []
for i in test:
    if i == 'Iris-setosa':
        class_label.append(0)
    elif i == 'Iris-virginica':
        class_label.append(2)
    else:
        class_label.append(1)

df = df.drop(['Species'], axis=1)                                                               #dropping 5th attribute as already labelled with vlaues ,1,2

# !1
pca = PCA(n_components=2)                                                                       #constructing the PCA model to reduce dimension
reduced_data = pd.DataFrame(pca.fit_transform(df), columns=['x1', 'x2'])                        #making a separate dataframe of reduced data to use in further questions
eigen_value, eigen_vector  = np.linalg.eig(df.corr().to_numpy())                                #function that gives eigen value and eigen vector
c = np.linspace(1, 4, 4)                                                            
plt.plot(c, [round(i,3) for i in eigen_value])                                                  #plotting the data points for eigen values of respective components
plt.xticks(np.arange(min(c), max(c)+1, 1.0))
plt.xlabel('Components')
plt.ylabel('Eigen Values')
plt.title('Eigen Values vs Components')
plt.show()


#Q2
K = 3                                                                                           # given
kmeans = KMeans(n_clusters=K)                                                                   #constructing K means model
kmeans.fit(reduced_data)                                                                        #provideing the data to the model and cluster it
k_pred = kmeans.predict(reduced_data)                                                           #predicting the data
reduced_data['k_cluster_label'] = kmeans.labels_                                                #making a sseparate column for the labels assigned to data points
k_center_coordinate = kmeans.cluster_centers_                                                   #stores the co-ordinates of the centre of cluster


plt.scatter(reduced_data[reduced_data.columns[0]], reduced_data[reduced_data.columns[1]],
            c=k_pred, cmap='rainbow', s=15)                                                     #scatter plot of the reduced cluster data
plt.scatter([k_center_coordinate[i][0] for i in range(K)], [k_center_coordinate[i][1] for i in range(K)],
            c='black', marker='o', label='cluster centre position')                             #scatter plot for the coordinates
plt.legend()
plt.title('K-Means')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

print('--------------------------------------------------------------------------------')
print('The distortion measure for k =3 is', round(kmeans.inertia_, 3))

def purity_score(y_true_value, y_predicted_value):                                              #fucntion to caluclate the purity score, will be used in further questions
    contingency_matrix = metrics.cluster.contingency_matrix(y_true_value, y_predicted_value)    #finding the condinjecny matrix
    row_index, column_index = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_index, column_index].sum() / np.sum(contingency_matrix)       #returning the purity score


print('---------------------------------------------------------------------------------')
print('The purity score is', round(purity_score(class_label, k_pred), 3))

# Q3
reduced_data = reduced_data.drop(['k_cluster_label'], axis=1)                                   #dropping the label from previous question to assign new labels
K = [2, 3, 4, 5, 6, 7]                                                                          #given  
k_disortion = []
purity = []
for k in K:
    kmeans = KMeans(n_clusters=k)                                                               #kmeans model for specific value of k
    kmeans.fit(reduced_data)                                                                    #fitting the data in that model
    k_disortion.append(round(kmeans.inertia_, 3))                                               #computing disortion measue
    purity.append(round(purity_score(class_label, kmeans.predict(reduced_data)), 3))            #computing purity score

print('----------------------------------------------------------------------------------')
print('The distortion measures are', k_disortion)
print('The purity scores are', purity)

plt.plot(K, k_disortion)                                                                        #plot for k vs disortion measure
plt.title('Distortion Measure vs K')
plt.xlabel('Value of K')
plt.ylabel('Distortion Measure')
plt.show()

# Q4
gmm = GaussianMixture(n_components=3, random_state=42).fit(reduced_data)                        #constructing the GMM Model
gmm_pred = gmm.predict(reduced_data)                                                            #predicting the clusters of the data
reduced_data['gmm_cluster_label'] = gmm_pred                                                    #making a separate column for the labels
gmm_centres_position = gmm.means_                                                               #coordinates of the centre

plt.scatter(reduced_data[reduced_data.columns[0]], reduced_data[reduced_data.columns[1]],
            c=gmm_pred, cmap='rainbow', s=15)                                                   #scatter plot for the data points in the clusters
plt.scatter([gmm_centres_position[i][0] for i in range(3)],
            [gmm_centres_position[i][1] for i in range(3)], c='black', marker='o',
            label='clusters centre position')                                                   #scatter plot for the coordinates of the centre
plt.legend()
plt.title('GMM Algorithm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

reduced_data = reduced_data.drop(['gmm_cluster_label'], axis=1)
print('-------------------------------------------------------------------------------')
print('The distortion measure for k = 3 is', round(gmm.score(reduced_data) * len(reduced_data), 3))             #disortion  measure 
print('The purity score for k =3 is', round(purity_score(class_label, gmm_pred), 3))            #purity score

# 5
total_log_probability = []
gmm_purity = []
for k in K:
    gmm = GaussianMixture(n_components=k, random_state=42).fit(reduced_data)                    #cosntructing gaussian mixutre model for different k
    total_log_probability.append(round(gmm.score(reduced_data) * len(reduced_data), 3))         #finding the total log probabilityt
    gmm_purity.append(round(purity_score(class_label, gmm.predict(reduced_data)), 3))           #finding the purity score

print('----------------------------------------------------------------------------')
print('The distortion measures are', total_log_probability)                                     #printing the disortion measure
print('The purity scores are', gmm_purity)                                                      #printing the purity score

plt.plot(K, total_log_probability)                                                              #plot for k and their disortion measure value
plt.title('Plot of Distortion Measure vs K')
plt.xlabel('K')
plt.ylabel('Distortion Measure')
plt.show()

# 6
print('------------------------------------------------------------------------------')
epsilon = [1, 1, 5, 5]                                                                          #epsilon values, given
minimum_samples = [4, 10, 4, 10]                                                                #minimum samples, given
for i in range(4):
    dbscan_model = DBSCAN(eps=epsilon[i], min_samples=minimum_samples[i]).fit(reduced_data)     #constucting the DBSCAN model for given values and fitting the data 
    DBSCAN_predictions = dbscan_model.labels_                                                   #finding the labels of predicted data points
    score = purity_score(class_label, DBSCAN_predictions)                                       #finding the purity score
    print(f'Purity score for epsilon={epsilon[i]} and minimum_samples={minimum_samples[i]} is',
          round(score, 3))                                                                      #printing purity score for given epsilon and min points
    plt.scatter(reduced_data[reduced_data.columns[0]], reduced_data[reduced_data.columns[1]],
                c=DBSCAN_predictions, cmap='flag', s=15)                                        #scatter plot for the data poiints
    plt.title(f'Data Points for epsilon={epsilon[i]} and minimum_samples={minimum_samples[i]}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
