#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:51:39 2022

@author: KHAN MAHMUDUL HASAN

 Python warm-up training to get used to simple manipulation and
 opertion of spyder operation.
 let's try to create the following simple math in the new file.
 before math statements, set-up library as follows;

"""
# library set-up
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split      # split data into train

# additional self-study part
# 6) examples for string, matrix, array, data frames
# define string
x = "Hello Great World"	# string data
print(x)
# define list
x1 = ["apple", "banana", "cherry"]	# list
print(x1)
# define tuple	
x2 = ("apple", "banana", "cherry")	# tuple
print(x2)
# define range	
x3 = range(6)	# range	
print(x3)
# define dictionary 
x4 = {"name" : "John", "age" : 36}	# dict
print(x4)
# define 2 x 3 matrix
arr1 = np.array([[1, 2, 3], [4, 5, 6]])  # by row order matrix
print(arr1)
# define 3 D array with 2 D matrix
arr2 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr2)
# define pandas data frame
df = pd.DataFrame({'a': [1, 2, 1, 3],
                   'b': [0.4, 1.1, 0.1, 0.8],
                   'c': ['X', 'Y', 'X', 'Z'],
                   'd': [[0, 0], [0, 1], [1, 0], [1, 1]]})
print(df)
#
# 7) for loop statement usage example
d=[1,2,3,4,5,6,7,8,9,10]  # define list data 
x =0  # define start value
for i in d:
    x = x + i  # loop 10 times (data count)
print(x)  # get sum data in d
print(x/len(d))  # calculate average
# 8) while loop statement usage example
# while is used conditional statement for looping
x = 0       # set initial value             
while x <= 5:  #   if x <= 5, execute below
   x = x + 1 
print(x)      # print x now, x is 6    
# end