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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split      # split data into train and test data
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
T = np.sum(d)     # define sum of d as T
print(T)     # print T
T
# caluculate mean (average) of d as np.mean(d)
ave =  np.mean(d)    # define ave for mean value 
print(ave)     # print ave
# check true or false for the following operations
if ave==5.0:
    print("True")
else :
    print("False")        # if ave is equal to 5.0
if ave>3:
    print("True")
else :
    print("False")      # if ave is  larger than 3
if ave>5:
    print("True")
else :
    print("False")         # if ave is larger than 5
if ave>7:
    print("True")
else :
    print("False")         # if ave is larger than 7
if ave<9:
    print("True")
else :
    print("False")         # if ave is smaller than 9
# 4) define character number list data as e
e=['1','2','3']  # character number data
# try to add each element and see what happened
T = np.char.add(e[0], np.char.add(e[1], e[2]))         # add element[0], e[1], e[2]
print(T)
# try to add each element and see what happened
# another way
strsum = "0"
sum = 0
for ch in e:
    if ch.isdigit():
        strsum += ch
        sum += int(ch)
print(int(strsum))
print(sum)
# 5) define tuple data as
f=(1,2,3,4)  # tuple data type and elements are immutable
# try to change element f[0] of tuple data to 5
         # try to change 1 to 5
#f[0] = 5 #TypeError: 'tuple' object does not support item assignment
print(f)
print(f[0])
ar = []
for num in f:
    ar.append(num)
ar[0] = 5
f = ar[0], ar[1], ar[2], ar[3]
print(f)
print(f[0])
#         

 
