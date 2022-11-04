# -*- coding: utf-8 -*-
"""
---
title: "PYR102 week5 assignment1"

@author: KHAN MAHMUDUL HASAN

---
PYR102 week5 assignment uses wine.csv data for PCA study.
 data outline:
 Number of Observations: 178 
 Number of Independent variables(X): V2 to V14 quantity data
 Dependent Variable(Y): V1 wine class data 

  Task assignment:
 1) read wine data from working directory
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
wine_data=pd.read_csv("wine.csv")  # get original data
# visualize covariate relationship
wine_data["V1"].astype(str)  # change to character numbers
xlist=wine_data.columns.to_list()[1:]  # get only quantity variable labels
sns.pairplot(wine_data, hue="V1", vars=xlist)   # plot by V1


"""
    
 2) standardize independent variables (x data) before pca

"""

# 2) standardize x data before PCA apply
wine_data1=wine_data.iloc[:,1:13]  # select quantity variable only
wine_data1s=preprocessing.scale(wine_data1) # standardize train data

"""

 3) Execute PCA function and get the following result data
    a) relationship data between variables and PCs
       and visualize them
       - scree plot for pc eigenvalues
       - get accumulative eigenvalue ratio
       - define major PCs (accumulative 70 - 80%) 
       - heat map for variable contribution to PCs

"""

# 3) execute PCA function
pca = PCA(svd_solver='randomized')
presult=pca.fit_transform(wine_data1s)  # get new variable data for reduction
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
           xticklabels=['V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13'],
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
dc.index=list(wine_data1.columns)  # set index labels
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

 4) define new x data using selected PC scores and y data
# Perfrom random forest classification using PCA result
 a) library set-up for classification
 b) divide data into learning data and test data
 Learning data size 80%, test data size 20%
 c) construct random forest model model using learning data
    set decision trees 100)
 d) fit model to train data 
    - set CV to 10
    - set scoring as accuracy
 e) get accuracy scores, mean, and standard deviation
 f) check if the case using less important PCs  
# end  

"""

# define new train data using PC scores
presult=pd.DataFrame(presult) # redefine as dataframe
# 4) Now, separate data into x data and y data
xpc=presult.iloc[:,0:5] # get 5 pc variables for x data 
y=wine_data.iloc[:,13]   # get y data
# 5) now split x,y data into train and test data
# test sample size 12%
x_train,x_test,y_train,y_test=train_test_split(xpc,y,test_size=0.20)

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
print("average deviation=",sum(diff)/len(diff))  # average of differences
print("max deviation=",max(diff))  # max difference

"""
Personal Practice : Linear Regression

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
wine_y_pred = regr.predict(x_test)
#
# Explained variance score: 1 is perfect prediction
print('Variance score: \n', r2_score(y_test, wine_y_pred))
# check prediction result
diff=abs(wine_y_pred-y_test)  # get absolute difference
print("average deviation=",sum(diff)/len(diff))  # average of differences
print("max deviation=",max(diff))  # max difference
# end
