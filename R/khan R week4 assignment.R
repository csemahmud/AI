# PYR101 week 4 task assignment
# @author: Student Name
# PYR101 week 4 task assignment
# this task will provide a basic training to do data preparation
# for machine learning input
# The followings are program steps you should create

# 1 ) Library set-up
#
library(MASS)
library(caret)  # get caret package
#
library(ggplot2)  # support for pair plot
library(reshape2)
#library(ggsci)
library(GGally)  # support for pair plot
library(dplyr)  # grouping function


# 2) read data from working directory (iris data.csv data) 
#
iris_data = read.csv("iris data.csv")
# Y data SPECIES	 
# X data SEPAL LENGTH, SEPAL WIDTH, PETAL LENGTH, PETAL WIDTH
#
# drop SPECIES column
iris_data1=iris_data[,c(2:5)]  # one way to drop SPECIES column
apply(is.na(iris_data1), 2, sum)  # check missing data
# or
x_iris_data=iris_data[, colnames(iris_data) != "SPECIES"]  # drop SPECIES
apply(is.na(x_iris_data), 2, sum)  # check missing data


# 3) Clean-up data to Drop or delete missing data (NaN) 
#
x_iris_data=na.omit(x_iris_data)  # drop NaN in dataframe 
rownames(x_iris_data)=c(1:nrow(x_iris_data))  # rename row label as ascending order
# Before dividing data, shuffle data
temp=sample(nrow(x_iris_data))  # shuffle row index data
x_iris_data=x_iris_data[temp,]  # reorder data by random vector 

# 4) Convert class label(SPECIES) if numeric to categorical data 
#    (character numbers) using the following;
#  data[,SPECIES"]=data[,"SPECIES"]=as.factor(data[,"SPECIES"])  # define as factor
#
iris_data[,"SPECIES"]=as.factor(iris_data[,"SPECIES"])  # define as factor
apply(is.na(x_iris_data), 2, sum)  # check missing data


# 5) Visualize variance relationship by SPECIES using ggpairs
apply(is.na(iris_data), 2, sum)  # check missing data
#
iris_data=na.omit(iris_data)  # drop NaN in dataframe 
rownames(iris_data)=c(1:nrow(iris_data))  # rename row label as ascending order
# Before dividing data, shuffle data
temp=sample(nrow(iris_data))  # shuffle row index data
iris_data=iris_data[temp,]  # reorder data by random vector
apply(is.na(iris_data), 2, sum)  # check missing data
# Pair plot for multivariate relationship analysis
ggpairs(iris_data,aes_string(colour="SPECIES", alpha=0.5))


# 6) divide data into train data and test data
# train data sample size 80% (test data 20%)
#

trainIndex = createDataPartition(iris_data$SPECIES, p = .80,
                                 list = FALSE,
                                 times = 1)
datatrain = x_iris_data[ trainIndex,]
datatest = x_iris_data[-trainIndex,]



