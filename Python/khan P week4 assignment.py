# -*- coding: utf-8 -*-
"""
PYR101 week 4 task assignment

@author: KHAN MAHMUDUL HASAN
"""
# PYR101 week 4 task assignment
# this task will provide a basic training to do data preparation
# for machine learning input
# The followings are program steps you should create
 
# 1 ) Library set-up
#
from sklearn.model_selection import train_test_split # split data into train and test data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 2) read data from working directory (iris data.csv data) 
# Y data SPECIES	 
# X data SEPAL LENGTH, SEPAL WIDTH, PETAL LENGTH, PETAL WIDTH
#
iris_data=pd.read_csv("iris data.csv")  # get original data
print(iris_data)  
print(iris_data.info()) # check data structure and missing data
# 3) Clean-up data to Drop or delete missing data (NaN) 
#
iris_data=iris_data.dropna()  # drop NaN in dataframe 
iris_data=iris_data.reset_index(drop=True) # reorder index
print(len(iris_data)) # get number of iris_data(row count)
# end

# drop missing data and define new dataframe as iris_data


iris_data.describe() # 4. get summary statistics by iris_data.describe()
# Before deviding data, shuffle data
iris_data=iris_data.sample(n=150,replace=False)  # shuffle data
# 4  Convert class label(SPECIES) if numeric to categorical data 
#    (character numbers) using the following;
#  data["SPECIES"].astype(str)  # change to character numbers
#
iris_data["SPECIES"]=iris_data["SPECIES"].astype(str)  # change to character numenric numbers
# 5) Visualize variance relationship by SPECIES using sns.pairplot
sns.pairplot(iris_data, hue="SPECIES")
plt.show()
# 6) select X data 4 length/width variables and Y as SPECIES
#
xtemp=iris_data.iloc[:,[1,2,3,4]]  # one way to drop SPECIES
x=iris_data.drop(columns=['SPECIES'])   # get x data except SPECIES
y=iris_data.iloc[:,0]   # get y data
# 7) split x, y data into train and test data
# test data 20%
#
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

