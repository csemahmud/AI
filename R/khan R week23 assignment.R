# PYR101 week 2 3 task assignment
# Author : KHAN MAHMUDUL HASAN
#
# this task will provide a basic training to manipulate
# and analyze data by using Glenco_data.csv in this holder. 
# 
# 1. library set-up
#
library(ggplot2)  # support for pair plot
library(reshape2)
#library(ggsci)
library(GGally)  # support for pair plot
library(dplyr)  # grouping function

Glenco_data = read.csv("Glenco_data.csv")
# 2. read Glenco dataframe as data from working directory 

apply(is.na(Glenco_data), 2, sum)
# 3. check data structure and missing data in dataframe
Glenco_data1=na.omit(Glenco_data)  # drop NaN in dataframe 
rownames(Glenco_data1)=c(1:nrow(Glenco_data1))  # rename row label as ascending order
print(nrow(Glenco_data1)) # get number of data1(row count)
# drop missing data and define new dataframe as data1

summary(Glenco_data1)
# 4. get summary statistics by summary(data)

# 5. Analyze Age data
# a) compare age average by gender using bar plot

g = ggplot(Glenco_data1, aes(x = Gender, y = Age))
g = g + geom_bar(stat = "identity")
plot(g) #Average Age

# b) compare age average by department using bar plot

g = ggplot(Glenco_data1, aes(x = Department, y = Age))
g = g + geom_bar(stat = "identity")
plot(g) #Average Age

# c) check age distribution for all using histogram
g = ggplot(Glenco_data1, aes(x = Age))
g = g + geom_histogram(position = "identity", alpha = 0.8)
plot(g)

# d) compare age distribution by Gender using histogram
g = ggplot(Glenco_data1, aes(x = Age, fill = Gender))
g = g + geom_density(aes(color = Gender, alpha = 0.2), show.legend = F)
g = g + geom_histogram(position = "identity", alpha = 0.8)
plot(g)

# 6. Analyze Tenure data
# a) check Tenure variance for all using box plot
g = ggplot(Glenco_data1, aes(y = Tenure))
g = g + geom_boxplot()
plot(g)

# b) compare Tenure variance by Gender using box plot
g = ggplot(Glenco_data1, aes(x = Gender, y = Tenure))
g = g + geom_boxplot()
plot(g)

# 7. Analyze Tenure and Hourly rate correlation
# a) check correlation for all using scatter plot with regression line
g = ggplot(Glenco_data1, aes(x = Tenure, y = HourlyRate))
g = g + geom_smooth(method = 'lm') + geom_point(aes(color = Tenure))
g = g + geom_smooth(method = 'lm')
plot(g)

# b) compare correlation by Gender using scatter plot with regression line
g = ggplot(Glenco_data1, aes(x = Tenure, y = HourlyRate, fill = Gender))
g = g + geom_smooth(method = 'lm') + geom_point(aes(color = Gender))
plot(g)
# end

###############################################

#Personal Experiment . Not Part of assignmentssssss

# 5. Analyze Age data

g = ggplot(Glenco_data1, aes(x = ID, y = Age, fill = Gender))
g = g + geom_bar(stat = "identity")
plot(g) #Details by ID

g = ggplot(Glenco_data1, aes(x = ID, y = Age, fill = Department))
g = g + geom_bar(stat = "identity")
plot(g) #Details by ID

################################################