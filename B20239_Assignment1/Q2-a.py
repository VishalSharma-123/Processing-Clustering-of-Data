# Name: Vishal Sharma
# Roll No: B20239


import pandas as pd                                     #importing panda module
from matplotlib import pyplot as plt                    #importing pyplot
df = pd.read_csv('pima-indians-diabetes.csv')           #accessing the file

df.drop(columns=['class'], inplace=True)                #dropping the column class, dont need it for this question

for column in df.columns:

    #finding all the attributes of data except Age, and giving them ylabel and distinct colors

    if (column != 'Age'):
        
        if (column == 'pregs'):
            a = 'No of times Pregnant'
            color = '#FF9AA2'
        if (column == 'plas'):
            a = 'Plasma glucose concentration'
            color = '#F7DF75'
        if (column == 'pres'):
            a = 'Diastolic blood pressure (mm Hg)'
            color = '#82E3F5'
        if (column == 'skin'):
            a = 'Triceps skin fold thickness (mm)'
            color = '#DA8FE9'
        if (column == 'test'):
            a = '2-Hour serum insulin (mu U/mL)'
        if (column == 'pedi'):
            color = '#EF801C'
            a = 'Diabetes pedigree function'
        if (column == 'BMI'):
            a = 'Body mass index (weight in kg/(height in m)^2)'
            color = '#00A0B0'
        
        plt.scatter(df['Age'], df[column], color = color)
        plt.xlabel('AGE (in years)')
        plt.ylabel(a)
        plt.show()