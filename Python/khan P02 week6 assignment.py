# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:13:31 2020

 PYR102 week 6 task assignment
 This task is to apply k-mean clustering to generate new label data
 and to apply logistic regression classification on quantity and 
 categorical independent variables
# 
 Task assignment:
 Library set-up
 1) read data file (diabetes.csv)
 2) select all independent variables(both quantity and categorical)
 3) perform k-mean clustering with 3 classes
 4) combine class result with data
 5) Before dividing data, shuffle data
 6) divide data into learning data and test data
#    Learning data size 80%, test data size 20%
 7) now separate x data and y data for learning and test data
    For x data, both categorical and quantity data
 8) construct logistic regression classification model using lerning data
    the function for logistic regression classification is;
 construct logistic regression classification model
 logit=LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial',max_iter=10000)

 logit.fit(x_train, y_train)  # set learning x and y data

 9) apply the model created for prediction under test data
 pred=logit.predict(x_test)  # set test data 

 10) evaluate the prediction result;
     - difference between predicted value and target value
     - get prediction accuracy %
# end  
"""

