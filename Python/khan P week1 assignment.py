# -*- coding: utf-8 -*-
"""
PYR101 week 1 task assignment

@author: KHAN MAHMUDUL HASAN 

# PYR101 week 1 task assignment

 Python warm-up training to get used to simple manipulation and
 opertion of spyder operation.
 let's try to create the following simple math in the new file.
 before math statements, set-up library as follows;

"""
# library set-up
import numpy as np
import pandas as pd
# 1) let's try to create a simple math formula

# define variables a=5, b=3 for formula y=a+b
a = 5         # define a
b = 3         # define b
# define simple formula y
y = a + b          # define y
a
b
y
# 2)  Present Value(PV) = future Value(FV) / (1 + Rate of Return(r))**n (period) 
# current cash 1000,000  r = 0.025 rate of return for 5 years
# calculate FV cash value after 5 years and print
pv = 1000000  # define PV 
r = 0.025  # define r interest rate
n = 5     # define n period
fv = pv*(1+r)**n     # define FV and get answer
print(fv)     # print FV print(FV)
# 3) define numeric list data or vector as d
d=[1,2,3,4,5,6,7,8,9,10]  # define list data type
# calculate sum of data d as sum(d)
T = sum(d)     # define sum of d as T
print(T)     # print T
# caluculate mean (average) of d as np.mean(d)
ave =  np.mean(d)    # define ave for mean value 
print(ave)     # print ave
# check true or false for the following operations
         # if ave is equal to 5.0
         # if ave is  larger than 3
         # if ave is larger than 5
         # if ave is larger than 7
         # if ave is smaller than 9
# 4) define character number list data as e
e=['1','2','3']  # character number data
# try to add each element and see what happened
         # add element[0], e[1], e[2]
# 5) define tuple data as
f=(1,2,3,4)  # tuple data type and elements are immutable
# try to change element f[0] of tuple data to 5
         # try to change 1 to 5
#         
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
 
