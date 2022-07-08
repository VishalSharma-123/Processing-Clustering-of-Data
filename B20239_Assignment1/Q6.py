# Name: Vishal Sharma
# Roll No: B20239


import pandas as pd                                     #importing panda module
from matplotlib import pyplot as plt                    #importing pyplot module
df = pd.read_csv('pima-indians-diabetes.csv')           #accessing the file

df.drop(columns=['class'], inplace=True)                #dropping the column class, dont need it for this question

#accessing all the attributes of the data
for column in df.columns:

    #for every attribute assigning the value of title for the box plot

    if (column == 'pregs'):
        a = 'No of times Pregnant'
    if (column == 'plas'):
        a = 'Plasma glucose concentration'
    if (column == 'pres'):
        a = 'Diastolic blood pressure (mm Hg)'
    if (column == 'skin'):
        a = 'Triceps skin fold thickness (mm)'
    if (column == 'test'):
        a = '2-Hour serum insulin (mu U/mL)'
    if (column == 'pedi'):
        a = 'Diabetes pedigree function'
    if (column == 'BMI'):
        a = 'Body mass index (weight in kg/(height in m)^2)'
    if (column == 'Age'):
        a = 'Age (years)'


    #using box plot function of pandas
    df.boxplot(column=column)
    plt.title(a)
    plt.show()