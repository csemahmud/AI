# -*- coding: utf-8 -*-
"""
PYR101 week 2 3 task assignment

@author: KHAN MAHMUDUL HASAN
"""

# PYR101 week 2 3 task assignment
# this task will provide a basic training to manipulate
# and analyze data by using Glenco_data.csv in this holder. 
# 
# 1. library set-up
import pandas as pd  # dataframe library
import numpy as np   # numeric library
import seaborn as sns  # graph library
import matplotlib.pyplot as plt  # graph package
#
Glenco_data=pd.read_csv("Glenco_data.csv") # 2. read Glenco dataframe form working directory 
print(Glenco_data.info())  # check data structure and missing data

# 3. check data structure and missing data in dataframe

# drop missin data in dataframe 
Glenco_data1=Glenco_data.dropna()  # drop NaN in dataframe 
Glenco_data1=Glenco_data1.reset_index(drop=True) # reorder index
print(len(Glenco_data1)) # get number of Glenco_data1(row count)
# end

# drop missing data and define new dataframe as Glenco_data1


Glenco_data1.describe() # 4. get summary statistics by df.describe()
 
# 5. Analyze Age data

sns.barplot(data=Glenco_data1, x= "Gender", y="Age")
# a) compare age average by gender using bar plot
plt.show()

sns.barplot(data=Glenco_data1, x= "ID", y="Age", hue='Gender')
# Details info by ID
plt.show()

sns.barplot(data=Glenco_data1, x= "Department", y="Age")
# b) compare age average by department using bar plot
plt.show()

sns.displot(data=Glenco_data1, x= "Age", kde=True)
# c) check age distribution for all using hisotgram
plt.show()

sns.displot(data=Glenco_data1, x= "Age", kde=True, hue='Gender')
# d) compare age distribution by Gender using histogram
plt.show()

# 6. Analyze Tenure data

ax = sns.boxplot(y="Tenure",data=Glenco_data1, linewidth=2.5)
# a) check Tenure variance for all using box plot
plt.show()

ax = sns.boxplot(y="Tenure", x="Gender",data=Glenco_data1, linewidth=2.5)
# b) compare Tenure variance by Gender using box plot
plt.show()

# 7. Analyze Tenure and Hourly rate correlation

sns.lmplot(x="Tenure", y="HourlyRate", data=Glenco_data1)
# a) check correlation for all using scatter plot with regression line
plt.show()

sns.lmplot(x="Tenure", y="HourlyRate", hue="Gender", data=Glenco_data1)
# b) compare correlation by Gender using scatter plot with regression line
plt.show()

# end

###############################################

#Personal Experiment . Not Part of assignment

sns.barplot(data=Glenco_data1, x= "ID", y="Age", hue='Department')
# Details info by ID
plt.show()

# 6. Analyze Tenure data
ax = sns.boxplot(data=Glenco_data1)
plt.show()

################################################
 