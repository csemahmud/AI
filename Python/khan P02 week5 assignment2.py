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
"""
    
 2) Perform correlation analysis among independent variables
 3) Execute PCA function and get the following result data
    a) relationship data between variables and PCs
       and visualize them
       - New variables defined as PCs (eigenvalues)
         and define major PCs (accumulative 70 - 80%) 
       - Variables and PCS relationship
       - Variable contribution to PCs
       - Variable cos2 distribution to PCs 

 4) Save PC scores with data in working directory for applying to regression
# Perfrom regression using PCA result
 Library set-up
 1) read data file (saved PCA data)
 2) Before dividing data, shuffle data
 3) divide data into learning data and test data
 Learning data size 400, test data size 106
 4) now separate x data and y data for learning and test data
    x data is the result based on standardized data
 5) construct linear regression model using learning data
 6) apply regression model created for prediction under test data
 7) evaluate the prediction result;
   - difference between predicted value and target value
   - average deviation from target value
   - maximum deviation
# end  

"""