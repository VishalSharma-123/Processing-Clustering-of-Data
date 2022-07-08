import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('landslide_data3_miss.csv')
df1 = pd.read_csv('landslide_data3_original.csv')

#--------------------------------------------------------------------------------------------

#Q1
a = list(df.columns)                                                                            #creating a list of attribute names
b = []

for i in df.columns:
    c = df[i].isna().sum()                                                                      #finiding no NA values in each attribute
    b.append(c)

plt.bar(a,b)                                                                                    #Plotting the bar graph
plt.ylabel('Number of missing values')
plt.title('No of missing valus in each  attribute')
plt.show()

#--------------------------------------------------------------------------------------------

#Q2 sub part a
a = df['stationid'].isnull().sum()                                                              #Finding no of NA values in attribute stationid
print ('\nNo of missing values in the attribute stationid is = {}\n'.format(a))

df.dropna(subset=['stationid'], inplace=True)                                                   #Dropping NA values from attribute stationid

index1 = list(df.index)                                                                         #Finding the indexes of missing and original csv file
index2 = list(df1.index)

def intersection(list1, list2):                                                                 #findinf the no of non common indexes from both data frames
    list3 = [value for value in list2 if value not in list1]
    return list3

index3 = intersection(index1, index2)                                           
df1.drop(index3, inplace=True)                                                                  #Dropping those non common indexes from original csv to maintain homogenity
df1 = df1.reset_index()                                                                         #resetiing the index to have a continuous value for both data frames
df = df.reset_index()

a = df.isnull().sum(axis=1)                                                                     #Finidng NA values 
count  = 0

#Q2 sub part b
for i in range(len(a)):                                                                         #Finidng attributes and deleting who have NA valeues > 3 in a tuple
    if (a[i] >= 3):
        df.drop([i], inplace=True)
        count +=1

print ('No of tuples having equal to or more than one third of attributes with missing values = {}\n'.format(count))
index1 = list(df.index)                                                                         #finding non common indexes, updatinf original data frame and reset index
index2 = list(df1.index)
index3 = intersection(index1, index2)
df1.drop(index3, inplace=True)
df1 = df1.reset_index()
df = df.reset_index()

#--------------------------------------------------------------------------------------------

#Q3
a = df.isnull().sum(axis=0)                                                                     #Finding the no of NA values in each attribute
print ('No of missing values in each column:\n')
print (a)
print ('\n')
print ('Total no of missing values after all these operations = {}\n'.format(a.sum()))

#--------------------------------------------------------------------------------------------

# Q4

a = df.copy()                                                                                   #Making a copy to use in further questions
b = df1.copy()

for columns in df.columns:

    #part a sub part 1

    if (df[columns].isna().sum() != 0):                                                         #Finding those columns who have NA valeus and replacing them with mean
        mean = df[columns].mean()
        df[columns].fillna(mean, inplace=True)

        print('For attribute {}:'.format(columns))                                              #Comparing mean, mode, median, std with original file
        print ('For the missing file')
        print ('Mean = {}'.format(df[columns].mean()))
        print ('Median = {}'.format(df[columns].median()))
        print ('Mode = {}'.format(df[columns].mode()[0]))
        print ('Standard Deviation = {}\n'.format(df[columns].std()))

        print ('For the original file')
        print ('Mean = {}'.format(df1[columns].mean()))
        print ('Median = {}'.format(df1[columns].median()))
        print ('Mode = {}'.format(df1[columns].mode()[0]))
        print ('Standard Deviation = {}'.format(df1[columns].std()))
        print ('---------------------------------------------------\n')

#Q4 part a sub part 2

def RMSE(a,b):                                                                                  #Creating a function to calculate RMSE
    rmse = 0
    for i in range(len(a)):
        rmse += (a[i] - b[i])**2
    
    return (rmse/len(a))**0.5

column = []
rmse = []
    
for columns in df.columns:
    if (columns != 'stationid' and columns != 'dates'):                                         #Excluding those as they are not numeric type
        column.append(columns)                                                                  #Filling list columns and corresponding RMSE values for bar plot
        rmse.append(RMSE(df[columns], df1[columns]))


del column[0]
del column[0]
del rmse[0]
del rmse[0]

plt.bar(column, rmse, log=True)                                                                 #Bar plot
plt.title('RMSE vs attributes')
plt.ylabel('RMSE')
plt.xlabel('ATTRIBUTES')
plt.show()

#--------------------------------------------------------------------------------------------

#Q4 part b sub part a

for columns in df.columns:

    if (a[columns].isna().sum() != 0):
        a[columns].interpolate(method='linear', limit_direction='forward', inplace=True)        #interpolating using interpolate function adn comparing ti with original 
        print('For attribute {}:'.format(columns))
        print ('For the missing file')
        print ('Mean = {}'.format(df[columns].mean()))
        print ('Median = {}'.format(df[columns].median()))
        print ('Mode = {}'.format(df[columns].mode()[0]))
        print ('Standard Deviation = {}\n'.format(df[columns].std()))

        print ('For the original file')
        print ('Mean = {}'.format(df1[columns].mean()))
        print ('Median = {}'.format(df1[columns].median()))
        print ('Mode = {}'.format(df1[columns].mode()[0]))
        print ('Standard Deviation = {}'.format(df1[columns].std()))
        print ('---------------------------------------------------\n')

#Q4 part b sub part 2

column = []                                                                                     #calculating RMSE values anf plotting a Bar Graph for it
rmse = []
    
for columns in df.columns:
    if (columns != 'stationid' and columns != 'dates'):
        column.append(columns)
        rmse.append(RMSE(a[columns], b[columns]))


del column[0]
del column[0]
del rmse[0]
del rmse[0]

plt.bar(column, rmse, log=True)
plt.title('RMSE vs attributes')
plt.ylabel('RMSE')
plt.xlabel('ATTRIBUTES')
plt.show()

#--------------------------------------------------------------------------------------------

Q1_temp = a['temperature'].quantile(q = 0.25)                                                   #fidning Q1, Q3 and IQR for temperature and rain attribute
Q3_temp = a['temperature'].quantile(q = 0.75)
Q1_rain = a['rain'].quantile(q = 0.25)
Q3_rain = a['rain'].quantile(q = 0.75)

iqr_temp = Q3_temp - Q1_temp
iqr_rain = Q3_rain - Q1_rain

upper_temp = Q3_temp + 1.5*iqr_temp                                                             #finding upper limit and lower limit of outliers
lower_temp = Q1_temp - 1.5*iqr_temp
upper_rain = Q3_rain + 1.5*iqr_rain
lower_rain = Q1_rain - 1.5*iqr_rain

outlier_temp = []
outlier_rain = []

for i in range(len(a['temperature'])):

    if (a['temperature'][i]>upper_temp or a['temperature'][i] < lower_temp):                    #Creating a list of values who dont lie in (upper - lower) range
        outlier_temp.append(i)

    if (a['rain'][i]>upper_rain or a['rain'][i] < lower_rain):
        outlier_rain.append(i)


print ('The list of outliers for the TEMPERATURE attribute is:\n{}'.format(outlier_temp))
print ('The list of outliers the RAIN attribute is:\n{}'.format(outlier_rain))

#Box Plot  
plt.boxplot(a['temperature'])                                                                   #Creating a Box Plot for it
plt.xlabel('TEMPERATURE')
plt.title('Box Plot of TEMPERATURE before replacing outliers with median')
plt.show()
plt.boxplot(a['rain'])
plt.yscale('log')
plt.title('Box Plot of RAIN before replacing outliers with median')
plt.xlabel('RAIN')
plt.show()

median1 = a['temperature'].median()
median2 = a['rain'].median()

for i in range(len(a['temperature'])):

    if (a['temperature'][i]>upper_temp or a['temperature'][i] < lower_temp):                    #Replacing outliers not in range (upper - lower) with median
        a.at[i, 'temperature'] = median1

    if (a['rain'][i]>upper_rain or a['rain'][i] < lower_rain):
        a.at[i, 'rain'] = median2

plt.boxplot(a['temperature'])                                                                   #Plotting a Box Plot of new values
plt.title('Box Plot of TEMPERATURE after replacing outliers with median')
plt.xlabel('TEMPERATURE')
plt.show()
plt.boxplot(a['rain'])
plt.yscale('log')
plt.title('Box Plot of RAIN after replacing outliers with median')
plt.xlabel('RAIN')
plt.show()

Q1_temp = a['temperature'].quantile(q = 0.25)                                                   #fidning Q1, Q3 and IQR for temperature and rain attribute
Q3_temp = a['temperature'].quantile(q = 0.75)
Q1_rain = a['rain'].quantile(q = 0.25)
Q3_rain = a['rain'].quantile(q = 0.75)

iqr_temp = Q3_temp - Q1_temp
iqr_rain = Q3_rain - Q1_rain

upper_temp = Q3_temp + 1.5*iqr_temp                                                             #finding upper limit and lower limit of outliers
lower_temp = Q1_temp - 1.5*iqr_temp
upper_rain = Q3_rain + 1.5*iqr_rain
lower_rain = Q1_rain - 1.5*iqr_rain


outlier_temp = []
outlier_rain = []

for i in range(len(a['temperature'])):

    if (a['temperature'][i]>upper_temp or a['temperature'][i] < lower_temp):                    #Creating a list of values who dont lie in (upper - lower) range
        outlier_temp.append(i)

    if (a['rain'][i]>upper_rain or a['rain'][i] < lower_rain):
        outlier_rain.append(i)

print (outlier_temp)
print (outlier_rain)