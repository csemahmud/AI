#' """
#' # PYR102 week 1 and 2 task assignment
#' @author: KHAN MAHMUDUL HASAN
#' # this task will provide a basic training to do data preparation
#' # for machine learning input
#' # about titanic_kaggle.csv data information (columns):
#' 
#' survival: Survival
#' PassengerId: Unique Id of a passenger.
#' pclass: Ticket class
#' sex: Sex
#' Age: Age in years
#' sibsp: # of siblings / spouses aboard the Titanic
#' Siblings mean brothers and sisters
#' parch: # of parents / children aboard the Titanic
#' ticket: Ticket number
#' fare: Passenger fare
#' cabin: Cabin number
#' embarked: Port of Embarkation(boarding)
#' Deck: cabin ID assigned from A to G 
#' who: man, woman, child identification
#' 
#'  The followings are task steps you should create
#' 
#' 1 ) Library set-up
#' """

library(MASS)
library(caret)  # get caret package
#
library(ggplot2)  # support for pair plot
library(reshape2)
#library(ggsci)
library(GGally)  # support for pair plot
library(dplyr)  # grouping function
library(fastDummies)  # load one-hot dummy creation 
library(superml)  # load label encoder program

# """
# 2) read data from working directory (titanic_kaggle.csv data) 
# 
# The following creates basic data information missing data
# """

titanic_kaggle_data = read.csv("titanic_kaggle.csv", na.strings=(c("NA", "")))
apply(is.na(titanic_kaggle_data), 2, sum)  # check missing data
summary(is.na.data.frame(titanic_kaggle_data))

# """
# 3) analyze variable relationship and trend by using graphs;
#  survived vs Pcalss, sex, age, sibsp, parch, fare,   
#  cabin, and embarked
# """

# d) compare Pcalss, sex, age, sibsp, parch, fare,   
#  cabin, and embarked distribution by Survived using histogram
g = ggplot(titanic_kaggle_data, aes(x = Pclass, fill = Survived))
g = g + geom_density(aes(color = Survived, alpha = 0.2), show.legend = F)
g = g + geom_histogram(stat = "count", position = "identity", alpha = 0.8)
plot(g)

g = ggplot(titanic_kaggle_data, aes(x = Sex, fill = Survived))
g = g + geom_density(aes(color = Survived, alpha = 0.2), show.legend = F)
g = g + geom_histogram(stat = "count", position = "identity", alpha = 0.8)
plot(g)

g = ggplot(titanic_kaggle_data, aes(x = Age, fill = Survived))
g = g + geom_density(aes(color = Survived, alpha = 0.2), show.legend = F)
g = g + geom_histogram(stat = "count", position = "identity", alpha = 0.8)
plot(g)

g = ggplot(titanic_kaggle_data, aes(x = SibSp, fill = Survived))
g = g + geom_density(aes(color = Survived, alpha = 0.2), show.legend = F)
g = g + geom_histogram(stat = "count", position = "identity", alpha = 0.8)
plot(g)

g = ggplot(titanic_kaggle_data, aes(x = Parch, fill = Survived))
g = g + geom_density(aes(color = Survived, alpha = 0.2), show.legend = F)
g = g + geom_histogram(stat = "count", position = "identity", alpha = 0.8)
plot(g)

g = ggplot(titanic_kaggle_data, aes(x = Fare, fill = Survived))
g = g + geom_density(aes(color = Survived, alpha = 0.2), show.legend = F)
g = g + geom_histogram(stat = "count", position = "identity", alpha = 0.8)
plot(g)

g = ggplot(titanic_kaggle_data, aes(x = Cabin, fill = Survived))
g = g + geom_density(aes(color = Survived, alpha = 0.2), show.legend = F)
g = g + geom_histogram(stat = "count", position = "identity", alpha = 0.8)
plot(g)

g = ggplot(titanic_kaggle_data, aes(x = Embarked, fill = Survived))
g = g + geom_density(aes(color = Survived, alpha = 0.2), show.legend = F)
g = g + geom_histogram(stat = "count", position = "identity", alpha = 0.8)
plot(g)


# """
# 4) By keeping total observation numbers, decide how to
#  to treate missing data (NA,NaN) in both categorical and numerical data
# """

titanic_kaggle_data1=na.omit(titanic_kaggle_data)  # drop NaN in dataframe 
rownames(titanic_kaggle_data1)=c(1:nrow(titanic_kaggle_data1))  # rename row label as ascending order
print(nrow(titanic_kaggle_data1)) # get number of data1(row count)
summary(titanic_kaggle_data1)

# """
# 5) Convert categorical data to numeric data
# """


# 1. Replacing values
# change carrier to numeric numbers
toPoint = function(factors) { 
  mapping =c('male'= 0, 'female'= 1, 
             'C'= 0, 'Q'= 1, 'S'= 2)
  
  mapping[as.character(factors)]
}

head(titanic_kaggle_data, n = 5)   # print 5 lines of data
str(titanic_kaggle_data)  # get data structure information
# define only categorical data (chr type)
cat_data = titanic_kaggle_data[,c("Sex", "Embarked")]
df=data.frame(cat_data)
cat_data = data.frame(lapply(df, toPoint))
head(cat_data, n = 5)
titanic_kaggle_data[,c("Sex", "Embarked")] = cat_data
head(titanic_kaggle_data, n = 5)

head(titanic_kaggle_data1, n = 5)   # print 5 lines of data
str(titanic_kaggle_data1)  # get data structure information
# define only categorical data (chr type)
cat_data = titanic_kaggle_data1[,c("Sex", "Embarked")]
df=data.frame(cat_data)
cat_data = data.frame(lapply(df, toPoint))
head(cat_data, n = 5)
titanic_kaggle_data1[,c("Sex", "Embarked")] = cat_data
head(titanic_kaggle_data1, n = 5)

# end

# """
# 6) Drop unnecesary columns if any
# """

# drop "Cabin",'PassengerId','Name','Ticket' column
titanic_kaggle_data = titanic_kaggle_data[, c(2,3,5,6,7,8,10,12)]  
# drop "Cabin",'PassengerId','Name','Ticket'
apply(is.na(titanic_kaggle_data), 2, sum)  # check missing data
head(titanic_kaggle_data, n = 5)

# """
# 7) Divide data into x data and y data (survived)
# """

titanic_kaggle_data=na.omit(titanic_kaggle_data)  # drop NaN in dataframe 
rownames(titanic_kaggle_data)=c(1:nrow(titanic_kaggle_data))  # rename row label as ascending order
print(nrow(titanic_kaggle_data)) # get number of data1(row count)
summary(titanic_kaggle_data)

# Y data Survived
#
# drop Survived column
x_titanic_kaggle_data=titanic_kaggle_data[, colnames(titanic_kaggle_data) != "Survived"]  # drop Survived
apply(is.na(x_titanic_kaggle_data), 2, sum)  # check missing data

# 3) # Before dividing data, shuffle data
temp=sample(nrow(x_titanic_kaggle_data))  # shuffle row index data
x_titanic_kaggle_data=x_titanic_kaggle_data[temp,]  # reorder data by random vector 

# 4) Convert class label(Survived) if numeric to categorical data 
#    (character numbers) using the following;
#  data[,Survived"]=data[,"Survived"]=as.factor(data[,"Survived"])  # define as factor
#
titanic_kaggle_data[,"Survived"]=as.factor(titanic_kaggle_data[,"Survived"])  # define as factor
apply(is.na(x_titanic_kaggle_data), 2, sum)  # check missing data

# 5) Before dividing data, shuffle data
temp=sample(nrow(titanic_kaggle_data))  # shuffle row index data
titanic_kaggle_data=titanic_kaggle_data[temp,]  # reorder data by random vector
apply(is.na(titanic_kaggle_data), 2, sum)  # check missing data
# Pair plot for multivariate relationship analysis
ggpairs(titanic_kaggle_data,aes_string(colour="Survived", alpha=0.5))

# Y data Survived
#
# drop Survived column
x_titanic_kaggle_data1=titanic_kaggle_data1[, colnames(titanic_kaggle_data1) != "Survived"]  # drop Survived
apply(is.na(x_titanic_kaggle_data1), 2, sum)  # check missing data

# 3) # Before dividing data, shuffle data
temp=sample(nrow(x_titanic_kaggle_data1))  # shuffle row index data
x_titanic_kaggle_data1=x_titanic_kaggle_data1[temp,]  # reorder data by random vector 

# 4) Convert class label(Survived) if numeric to categorical data 
#    (character numbers) using the following;
#  data[,Survived"]=data[,"Survived"]=as.factor(data[,"Survived"])  # define as factor
#
titanic_kaggle_data1[,"Survived"]=as.factor(titanic_kaggle_data1[,"Survived"])  # define as factor
apply(is.na(x_titanic_kaggle_data1), 2, sum)  # check missing data

# 5) Before dividing data, shuffle data
temp=sample(nrow(titanic_kaggle_data1))  # shuffle row index data
titanic_kaggle_data1=titanic_kaggle_data1[temp,]  # reorder data by random vector
apply(is.na(titanic_kaggle_data1), 2, sum)  # check missing data

# """
# 8) split x, y data into train and test data
#   ( test data 20% )
# #
# """

trainIndex = createDataPartition(titanic_kaggle_data$Survived, p = .80,
                                 list = FALSE,
                                 times = 1)
datatrain = titanic_kaggle_data[ trainIndex,]
datatest = titanic_kaggle_data[-trainIndex,]

trainIndex1 = createDataPartition(titanic_kaggle_data1$Survived, p = .80,
                                 list = FALSE,
                                 times = 1)
datatrain1 = titanic_kaggle_data1[ trainIndex1,]
datatest1 = titanic_kaggle_data1[-trainIndex1,]


