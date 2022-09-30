# -*- coding: utf-8 -*-
"""
# PYR102 week 1 and 2 task assignment

@author: KHAN MAHMUDUL HASAN

# this task will provide a basic training to do data preparation
# for machine learning input
# about titanic_kaggle.csv data information (columns):

survived: 1 for Survival, 0 for non survival
PassengerId: Unique Id of a passenger.
pclass: Ticket class
sex: Sex
Age: Age in years
sibsp: # of siblings / spouses aboard the Titanic
Siblings mean brothers and sisters
parch: # of parents / children aboard the Titanic
ticket: Ticket number
fare: Passenger fare
cabin: Cabin number (A to G)
embarked: Port of Embarkation(boarding)
  C＝Cherbourg, Q＝Queenstown, S＝Southampton 

 The followings are task steps you should create

1 ) Library set-up

"""
from sklearn.model_selection import train_test_split # split data into train and test data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""    
2) read data from working directory (titanic_kaggle.csv data) 

The following creates basic data information missing data
"""

titanic_kaggle_data = pd.read_csv('titanic_kaggle.csv')
print(titanic_kaggle_data)
print(titanic_kaggle_data.info())
total = titanic_kaggle_data.isnull().sum().sort_values(ascending=False)
percent_1 = titanic_kaggle_data.isnull().sum()/titanic_kaggle_data.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total',
'%'])
missing_data.head(5)

"""
3) analyze variable relationship and trend by using graphs;
 survived vs Pcalss, sex, age, sibsp, parch, fare,   
 cabin, and embarked

"""
sns.pairplot(titanic_kaggle_data, hue="Survived")
plt.show()

# Useful graphs for analysis
# ploting for overall analysis
sns.pairplot(titanic_kaggle_data, hue='Survived')
#
# Plot histograms/bar plot for each feature and showing Survived
sns.displot(titanic_kaggle_data, x='Pclass', hue='Survived', multiple='dodge')  
# 
sns.displot(titanic_kaggle_data, x='Sex', hue='Survived', multiple='dodge')     
# 
sns.displot(titanic_kaggle_data, x='Age', hue='Survived', multiple='dodge') 
# 
sns.displot(titanic_kaggle_data, x='SibSp', hue='Survived', multiple='dodge')   
# 
sns.displot(titanic_kaggle_data, x='Parch', hue='Survived', multiple='dodge')   
#
sns.displot(titanic_kaggle_data, x='Fare', hue='Survived', multiple='dodge', bins=8)    
#
sns.displot(titanic_kaggle_data, x='Survived', hue='Cabin', multiple='dodge')    
#
sns.displot(titanic_kaggle_data, x='Survived', hue='Embarked', multiple='dodge') 
#  
# extra graph
plt.figure(figsize=(8,5))
sns.boxplot(x='Pclass',y='Age',data=titanic_kaggle_data, palette='rainbow', hue='Survived')
plt.title("Age by Passenger Class, Titanic")
sns.boxplot(x='Pclass',y='Age',data=titanic_kaggle_data, hue='Sex') 
#

"""

4) By keeping total observation numbers, decide how to
 to treate missing data (NA,NaN) in both categorical and numerical data

"""

#
titanic_kaggle_data1=titanic_kaggle_data.dropna()  # drop NaN in dataframe 
titanic_kaggle_data1=titanic_kaggle_data1.reset_index(drop=True) # reorder index
print(len(titanic_kaggle_data1)) # get number of iris_data(row count)
# end

titanic_kaggle_data1.describe()

"""

5) Convert categorical data to numeric data

"""    

"""

#
# Convert categorical data to numeric data
    # Note: convert 'Embarked' and 'Sex' to dummy variables, also for 'Pclass'
data1 = pd.get_dummies(titanic_kaggle_data, columns=['Pclass','Embarked','Sex'])
#
# calculate average age per Sex and Pclass
meanvalue = data1.groupby(['Pclass', 'Sex'])['Age'].transform('mean')

Here the following error occurs :
    data1 = pd.get_dummies(titanic_kaggle_data, columns=['Pclass','Embarked','Sex'])
    #
    # calculate average age per Sex and Pclass
    meanvalue = data1.groupby(['Pclass', 'Sex'])['Age'].transform('mean')
    Traceback (most recent call last):

      File "/tmp/ipykernel_2524453/1610192227.py", line 4, in <module>
        meanvalue = data1.groupby(['Pclass', 'Sex'])['Age'].transform('mean')

      File "/root/anaconda3/lib/python3.9/site-packages/pandas/core/frame.py", line 7712, in groupby
        return DataFrameGroupBy(

      File "/root/anaconda3/lib/python3.9/site-packages/pandas/core/groupby/groupby.py", line 882, in __init__
        grouper, exclusions, obj = get_grouper(

      File "/root/anaconda3/lib/python3.9/site-packages/pandas/core/groupby/grouper.py", line 882, in get_grouper
        raise KeyError(gpr)

    KeyError: 'Pclass'

data1['Age'] = data1['Age'].fillna(meanvalue)
#
# Filling randomly selected ports in NAN
freq_c = data1[data1['Embarked'] =='C'].Embarked.count() / data1.Embarked.count()
freq_s = data1[data1['Embarked'] =='S'].Embarked.count() / data1.Embarked.count()
freq_q = data1[data1['Embarked'] =='Q'].Embarked.count() / data1.Embarked.count()

portlist = ['C', 'S', 'Q']
problist = [freq_c, freq_s, freq_q]
# randomly choose based on ports with probability
data1['Embarked'] = data1['Embarked'].fillna(pd.Series(np.random.choice(portlist,p=problist,replace=True,size=len(data1.index))))

"""

# define replace value map
replace_map = {'Sex': {'male': 0, 'female': 1},
               'Embarked': {'C': 0, 'Q': 1, 'S': 2}}

titanic_kaggle_data.replace(replace_map, inplace=True) # change to numeric
print(titanic_kaggle_data["Sex"], titanic_kaggle_data["Embarked"])
print(titanic_kaggle_data.info())


titanic_kaggle_data1.replace(replace_map, inplace=True) # change to numeric
print(titanic_kaggle_data1["Sex"], titanic_kaggle_data1["Embarked"])
print(titanic_kaggle_data1.info())
"""

6) Drop unnecesary columns if any

"""
#Dropping "Cabin",'PassengerId','Name','Ticket'

titanic_kaggle_data = titanic_kaggle_data.drop(columns=["Cabin",'PassengerId','Name','Ticket']);
print(titanic_kaggle_data)
print(titanic_kaggle_data.info())

# drop non required columns

"""
    
7) Divide data into x data and y data (survived)

"""
#
titanic_kaggle_data=titanic_kaggle_data.dropna()  # drop NaN in dataframe 
titanic_kaggle_data=titanic_kaggle_data.reset_index(drop=True) # reorder index
print(len(titanic_kaggle_data)) # get number of iris_data(row count)
# end

titanic_kaggle_data.describe()

titanic_kaggle_data["Survived"]=titanic_kaggle_data["Survived"].astype(str)  # change to character numenric numbers
print(titanic_kaggle_data)
print(titanic_kaggle_data.info())

# Without Cabin

x=titanic_kaggle_data.iloc[:,[0,2,3,4,5,6,7,8,9,10]]  # one way to drop Survived
y=titanic_kaggle_data.iloc[:,1]   # get y data

# With Cabin

titanic_kaggle_data1["Survived"]=titanic_kaggle_data1["Survived"].astype(str)  # change to character numenric numbers
print(titanic_kaggle_data1)
print(titanic_kaggle_data1.info())

x1=titanic_kaggle_data1.iloc[:,[0,2,3,4,5,6,7,8,9,10,11]]  # one way to drop Survived
y1=titanic_kaggle_data1.iloc[:,1]   # get y data

"""

8) split x, y data into train and test data
  ( test data 20% )
#
"""
# Without Cabin

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# With Cabin

x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2)

"""
Self Practice using numpy library

"""

# calculate sum of Fare as sum(titanic_kaggle_data1["Fare"])
sum = np.sum(titanic_kaggle_data1["Fare"])     # define sum of Fare as sum
print(sum)     # print sum
sum
# caluculate mean (average) of Age as np.mean(titanic_kaggle_data1["Age"])
ave =  np.mean(titanic_kaggle_data1["Age"])    # define ave for mean value 
print(ave)     # print ave

