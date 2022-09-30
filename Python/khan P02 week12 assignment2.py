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

import pandas as pd

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
# end
