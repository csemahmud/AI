# -*- coding: utf-8 -*-
"""
---
title: "PYR102 week5 assignment2"

@author: KHAN MAHMUDUL HASAN

---
PYR102 week5 assignment uses Boston House Price.csv data for PCA study.
 data outline:
 Number of Observations: 506 
 Number of Independent variables(X): 11 quantity/2 categorical data
 Dependent Variable(Y): MEDV 

 Variable Information:
 Quantity variables:
 - MEDV     Median value of owner-occupied homes in $1000's
 - CRIM     per capita crime rate by town
 - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 - INDUS    proportion of non-retail business acres per town
 - NOX      nitric oxides concentration (parts per 10 million)
 - RM       average number of rooms per dwelling
 - AGE      proportion of owner-occupied units built prior to 1940
 - DIS      weighted distances to five Boston employment centres
 - TAX      full-value property-tax rate per $10,000
 - PTRATIO  pupil-teacher ratio by town
 - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 - LSTAT    % lower status of the population
 Categorical variables:
 - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 - RAD      index of accessibility to radial highways
 
 Task assignment:
 1) read Boston House Price data from working directory
    and select quantity independent variables only
    
"""

# New PYR102 PCA and regession study
# Apply  PCA model for understanding variavle contribution
# to principal components
# Library set-up
import pandas as pd
#import numpy as np
import seaborn as sns
import random as rnd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing  # get standardize program
import warnings
warnings.simplefilter('ignore')


from sklearn import linear_model
from sklearn.metrics import r2_score
#import pandas as pd
#from sklearn.decomposition import PCA
#from sklearn import preprocessing  # get standardize program
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split # split data into train and test data


# 1) read data from working directory (diabetes data) 
boston_data=pd.read_csv("Boston House Price.csv")  # get original data

# 2) standardize x data before PCA apply
boston_data1=boston_data.iloc[:,0:12]  # select quantity variable only
boston_data1s=preprocessing.scale(boston_data1) # standardize train data

"""
    
 2) Perform correlation analysis among independent variables
    
"""

# visualize covariate relationship
boston_data["MEDV"].astype(str)  # change to character numbers
xlist=boston_data.columns.to_list()[1:]  # get only quantity variable labels
sns.pairplot(boston_data, hue="MEDV", vars=xlist)   # plot by MEDV

""" 

 3) Execute PCA function and get the following result data
    a) relationship data between variables and PCs
       and visualize them
       - New variables defined as PCs (eigenvalues)
         and define major PCs (accumulative 70 - 80%) 
       - Variables and PCS relationship
       - Variable contribution to PCs
       - Variable cos2 distribution to PCs 

"""

# 3) execute PCA function
pca = PCA(svd_solver='randomized')
presult=pca.fit_transform(boston_data1s)  # get new variable data for reduction
#print(presult)  # pca data for all observation
print('PC factor loadings:')
print(pca.components_)  # print factor loading
ev = pca.explained_variance_   # get eigenvalue
print('Eigenvalues=')
print(ev)
pw=pd.DataFrame(ev)  # define as data frame
plt.figure()
plt.plot(ev, 's-')   # plot screen chart
plt.title("Scree chart")
plt.xlabel("factors")
plt.ylabel("eigenvalue")
plt.grid(True)
plt.show()
print('explained variance ratio %')
print(pca.explained_variance_ratio_) # eigenvalue accumurative %
print('accumurative eigenvalue ratio')
print(pca.explained_variance_ratio_.cumsum())  # print accumurative %

# plot heat map to analyze variable contribution to PCs
v=pca.components_   # get PC components
fig, ax = plt.subplots(figsize=(8, 8))
plt.grid(False)
sns.heatmap(v,
           linewidths=.5,  # divide each cell by white line
           ax=ax,          # use subplot ax
           cmap='Blues',   # heat map color blue
           annot=True,     # froating numbers in cell
           annot_kws={'size': 14},
           fmt='.2f',      # number format as froating
           xticklabels=['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','TAX','PTRATIO','B','LSTAT'],
           yticklabels=['PC1', 'PC2', 'PC3', 'PC4','PC5','PC6','PC7','PC8','PC9'])
ax.set_ylim(len(v), 0)  # adjust y axes limits
plt.title("Variable contribution to PCs")
plt.show()

# get factor loading graph
plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 15
# plot pca factor loadings
pv=pd.DataFrame(v)  # redefine rotation as dataframe
pvt=pv.T  # transpose dataframe
dc=pvt.iloc[:,[0,1]] # get eigenvector for PC1 and PC2
a=(pw.iloc[0,0])**0.5   # square root eigenvalue pc1
b=(pw.iloc[1,0])**0.5   # square root eigenvalue pc2
dc.iloc[:,0]=(dc.iloc[:,0])*a # factor loading pc1
dc.iloc[:,1]=(dc.iloc[:,1])*b # factor loading pc2
dc.index=list(boston_data1.columns)  # set index labels
print(dc)
plt.scatter(dc[0], dc[1], s=100, marker="x")
# print labels for each plot
cnt = 0
for label in list(dc.index):
    r = rnd.random() * 0.1
    plt.text(dc.iloc[cnt, 0]+r, dc.iloc[cnt, 1]+r, label)
    plt.plot([dc.iloc[cnt, 0]+r,dc.iloc[cnt, 0]], [dc.iloc[cnt, 1]+r, dc.iloc[cnt, 1]])
    plt.plot([0,dc.iloc[cnt, 0]], [0, dc.iloc[cnt, 1]],color="red")
    cnt += 1
plt.title("PCA Factor loading chart")
plt.xlabel("factor 1 (pc1)")
plt.ylabel("factor 2 (pc2)")
plt.grid(True)
plt.show()
# end

"""

 4) Save PC scores with data in working directory for applying to regression
# Perfrom regression using PCA result
 Library set-up
 1) read data file (saved PCA data)
 2) Before dividing data, shuffle data
 3) divide data into learning data and test data
 Learning data size 400, test data size 106 
 4) now separate x data and y data for learning and test data
    x data is the result based on standardized data
 
"""

# define new train data using PC scores
presult=pd.DataFrame(presult) # redefine as dataframe
# 4) Now, separate data into x data and y data
xpc=presult.iloc[:,0:5] # get 5 pc variables for x data 
y=boston_data.iloc[:,12:14]   # get y data
# 5) now split x,y data into train and test data
# test sample size 12%
x_train,x_test,y_train,y_test=train_test_split(xpc,y,test_size=0.209)


"""

 5) construct linear regression model using learning data
 6) apply regression model created for prediction under test data
 7) evaluate the prediction result;
   - difference between predicted value and target value
   - average deviation from target value
   - maximum deviation
# end  

"""

# 6) execute K devision cross validation
# split train data into 5 with shuffle option
kfold_cv = KFold(n_splits=5, shuffle=True)  
# set regressor
regr = linear_model.LinearRegression()
# execute cross validation
scores = cross_val_score(regr, x_train, y_train, cv=kfold_cv)
print("accuracy % = ")
print(scores)
#
# Train the model using the training sets
regr.fit(x_train, y_train)

# 7) Make predictions using the testing set
boston_y_pred = regr.predict(x_test)
#
# Explained variance score: 1 is perfect prediction
print('Variance score: \n', r2_score(y_test, boston_y_pred))
# check prediction result
diff=abs(boston_y_pred-y_test)  # get absolute difference
print("average deviation of CHAS=",sum(diff["CHAS"])/len(diff["CHAS"]))  # average of differences
print("max deviation of CHAS=",max(diff["CHAS"]))  # max difference
print("average deviation of RAD=",sum(diff["RAD"])/len(diff["RAD"]))  # average of differences
print("max deviation of RAD=",max(diff["RAD"]))  # max difference
# end

"""
Personal Practice : Ramdom Forest Regression model

"""

# 3) Ramdom Forest Regression model
# Construct Random forest model
from sklearn.ensemble import RandomForestRegressor
#
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_train, y_train)
pre_y = rfr.predict(x_test)
# Explained variance score: 1 is perfect prediction
print('Variance score: \n', r2_score(y_test, pre_y))
# check prediction result
print(pre_y-y_test)  # difference between prediction data and test data
diff=abs(pre_y-y_test)  # get absolute difference
print("average deviation of CHAS=",sum(diff["CHAS"])/len(diff["CHAS"]))  # average of differences
print("max deviation of CHAS=",max(diff["CHAS"]))  # max difference
print("average deviation of RAD=",sum(diff["RAD"])/len(diff["RAD"]))  # average of differences
print("max deviation of RAD=",max(diff["RAD"]))  # max difference
