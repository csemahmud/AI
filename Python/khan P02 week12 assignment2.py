# -*- coding: utf-8 -*-
"""
---
title: "R week12 assighment 2"

@author: KHAN MAHMUDUL HASAN

---
This PYR102 task is to prepare time series data for 
regression by using sliding window 
1. Read time series data (toyota_stock.csv).

"""

# PYR102 regression sample for time series data

# Library set-up
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.svm import SVR
#from sklearn.model_selection import train_test_split # split data into train and test data
import numpy as np

toyota_stock_data = pd.read_csv('toyota_stock.csv', encoding="utf-8")
# reorder data from old to new date
toyota_stock_data = toyota_stock_data.sort_index(axis=0, ascending=False)
#
print(toyota_stock_data)
print(toyota_stock_data.info())

"""
2. data summary
time series data: Toyota_stock_csv
period: 2021 4/16 to 2020 4/20
window interval: 6 (day1 to day6)
target: day7 ()End stock price)
training data: 2020 data
test data: 2021 data 

"""

# Define training and test data
train = (toyota_stock_data["Year"] <= 2020)
test = (toyota_stock_data["Year"] >= 2021)
interval = 6


"""
3.Data interval is 6 days time series and day 7th data 
is as Y data.
4. For regression, the day1 to day6 is defined as independent
 variables, day7 as dependent variable.
"""

# define sliding window (past 6 days) function
def make_data(data):
    x = [] # train data
    y = [] # train target data
    ends = list(data["End"])  # get Temperature data
    for i in range(len(ends)):  
        if i < interval: continue
        y.append(ends[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(ends[d])
        x.append(xa)
    return (x, y)

"""
5. Create train and test data by using sliding window 
method.
6. Separate both train and test data into x data and y data.

"""

# define training y and x variables
train_x, train_y = make_data(toyota_stock_data[train])  # define train data
test_x, test_y = make_data(toyota_stock_data[test])  # define test data
#
#
# perform linear regression 
lr = LinearRegression(normalize=True)
lr.fit(train_x, train_y) # fit learning
pre_y = lr.predict(test_x) # prediction for test data

# The coefficients
print('Coefficients: \n', lr.coef_)
# Explained variance score: 1 is perfect prediction
print('Variance score: \n', r2_score(test_y, pre_y))

# plot test data Y and projected Y
plt.figure(figsize=(10,6))  # set figure size 6 x 10
plt.plot(test_y, color = 'black', linestyle = 'solid')
plt.plot(pre_y, color = 'red', linestyle = 'dotted')
plt.title('Test Y and Projected Y difference')
plt.xlabel("Observation")
plt.ylabel("day7 ()End stock price")
plt.show()

print(pre_y-test_y)  # difference between prediction data and test data
diff_y=abs(pre_y-test_y)
print("average deviation=",sum(diff_y)/len(diff_y))  # average of differences
print("max deviation=",max(diff_y))  # max difference
#
#
# 2) SVM model
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
#
svrs = [ svr_lin, svr_rbf]
for svr in svrs:
    svr.fit(train_x, train_y)
   # Make predictions using the testing set
    pre_y = svr.predict(test_x)
   #
    print(svr)
   # Explained variance score: 1 is perfect prediction
    print('Variance score: \n', r2_score(test_y, pre_y))
   # check prediction result
   # print(diabetes_y_pred-y_test)  # difference between prediction data and test data
    diff=abs(pre_y - test_y)  # get absolute difference
    print("average deviation=",sum(diff)/len(diff))  # average of differences
    print("max deviation=",max(diff))  # max difference
#

# 3) Ramdom Forest Regression model
# Construct Random forest model
from sklearn.ensemble import RandomForestRegressor
#
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(train_x, train_y)
pre_y = rfr.predict(test_x)
# Explained variance score: 1 is perfect prediction
print('Variance score: \n', r2_score(test_y, pre_y))
# check prediction result
print(pre_y-test_y)  # difference between prediction data and test data
diff=abs(pre_y-test_y)  # get absolute difference
print("average deviation=",sum(diff)/len(diff))  # average of differences
print("max deviation=",max(diff))  # max difference


# relative importance
train_x = pd.DataFrame(train_x)
importances = pd.DataFrame({'feature':train_x.columns,'importance':np.round(rfr.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.plot.bar()

# end
