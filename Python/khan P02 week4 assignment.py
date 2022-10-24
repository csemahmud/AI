# -*- coding: utf-8 -*-
"""
---
title: "PYR102 R week4 assignment"

@author: KHAN MAHMUDUL HASAN

---
This week4 assignment is a continuation from week12 assignment and is to create regression models and evaluation. 
The spitted x, y data into train and test data are now 
prepared ( test data 20% ) and to be used for the following
assignment.
# 
# week4 assignment:(data is toyota_stock.csv)
1) Library set-up for regression

"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split # split data into train and test data
import numpy as np

"""

2) construct regression models using linear regression model and  RandomForest model.

"""

toyota_stock_data = pd.read_csv('toyota_stock.csv', encoding="utf-8")
toyota_stock_data = toyota_stock_data.sort_index(axis=0, ascending=False)
#
print(toyota_stock_data)
print(toyota_stock_data.info())

train = (toyota_stock_data["Year"] <= 2020)
test = (toyota_stock_data["Year"] >= 2021)
interval = 6

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

train_x, train_y = make_data(toyota_stock_data[train])  # define train data
test_x, test_y = make_data(toyota_stock_data[test])  # define test data

svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

    
# Construct Random forest model
from sklearn.ensemble import RandomForestRegressor
#
rfr = RandomForestRegressor(n_estimators=100)


"""

3) apply created models for prediction under test data

"""


lr = LinearRegression(normalize=True)
lr.fit(train_x, train_y) # fit learning
pre_y = lr.predict(test_x) # prediction for test data

print('Coefficients: \n', lr.coef_)
print('Variance score: \n', r2_score(test_y, pre_y))

plt.figure(figsize=(10,6))  # set figure size 6 x 10
plt.plot(test_y, color = 'black', linestyle = 'solid')
plt.plot(pre_y, color = 'red', linestyle = 'dotted')
plt.title('Test Y and Projected Y difference')
plt.xlabel("Observation")
plt.ylabel("day7 ()End stock price")
plt.show()
#

svrs = [ svr_lin, svr_rbf]
for svr in svrs:
    svr.fit(train_x, train_y)
   # Make predictions using the testing set
    pre_y_svr = svr.predict(test_x)
   #
    print(svr)
   

rfr.fit(train_x, train_y)
pre_y_rfr = rfr.predict(test_x)


"""

4) evaluate the prediction result;
   - difference between predicted quantity and target quantity.
   - prediction accuracy
   - comparison between RandomForest model and linear regression model 
# end

"""

print(pre_y-test_y)  # difference between prediction data and test data
diff_y=abs(pre_y-test_y)
print("average deviation=",sum(diff_y)/len(diff_y))  # average of differences
print("max deviation=",max(diff_y))  # max difference

# Explained variance score: 1 is perfect prediction
print('Variance score: \n', r2_score(test_y, pre_y_svr))
# check prediction result
# print(diabetes_y_pred-y_test)  # difference between prediction data and test data
diff_svr=abs(pre_y_svr - test_y)  # get absolute difference
print("average deviation=",sum(diff_svr)/len(diff_svr))  # average of differences
print("max deviation=",max(diff_svr))  # max difference

# Explained variance score: 1 is perfect prediction
print('Variance score: \n', r2_score(test_y, pre_y_rfr))
# check prediction result
print(pre_y_rfr-test_y)  # difference between prediction data and test data
diff_rfr=abs(pre_y_rfr-test_y)  # get absolute difference
print("average deviation=",sum(diff_rfr)/len(diff_rfr))  # average of differences
print("max deviation=",max(diff_rfr))  # max difference

importances = pd.DataFrame({'feature':train_x.columns,'importance':np.round(rfr.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.plot.bar()

