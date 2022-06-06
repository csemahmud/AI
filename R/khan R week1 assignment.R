# PYR101 week 1 task assignment
# Author : KHAN MAHMUDUL HASAN
#
# R warm-up training to get used to simple manipulation and
# operation of Rstudio.
# let's try to create the following simple math in the new file.
#
# 1) let's try to create a simple math (5+3)*5-(40/2)
answer = (5+3)*5-(40/2)       # define math as answer
print(answer)       # print answer print(answer)
# 2) define variables a=5, b=3 and formula y=a+b
a=5       # define a
b=3       # define b
y=a+b       # define simple formula y
print(y)       # print y
# 3) Present Value(PV) = future Value(FV) / (1 + Rate of Return(r))^n (period) 
# current cash 1000,000  r = 0.025 5 years
# calculate FV cash value after 5 years and print
pv = 1000000       # define PV
r = 0.025       # define r
n = 5       # define n
fv = pv*(1+r)**n       # define FV and get value
print(fv)       # print FV
#
# 4) define list data or vector as d 
d=c(1,2,3,4,5,6,7,8,9,10)  # combination
# calculate sum of data d  sum(d) as T
T = sum(d)       # define T as sum(d)
print(T)       # print T
# 5) calculate mean (average) of d mean(d) as ave 
ave = mean(d)       # define ave mean(d)
print(ave)       # print ave
# check true or false for the following operations
if(ave==5.0){
  print("True")
}else{
  print("False")
}      # if ave equal to 5.0
if(ave>3){
  print("True")
}else{
  print("False")
}       # if ave is larger than 3
if(ave>5){
  print("True")
}else{
  print("False")
}       # if ave is larger than 5
if(ave>7){
  print("True")
}else{
  print("False")
}       # if ave is larger than 7
if(ave<9){
  print("True")
}else{
  print("False")
}       # if ave is smaller than 9
# 6) define character number as e vector
e=c('1','2','3')  # character number list data
# try to add each element and see what happened
T = paste(e[0], e[1], e[2], sep="")       # add element e[0], e[1], e[2]
print(T)
T = paste(T, e[3], sep="")       # add T with element e[3]
print(T)
# try to add each element and see what happened
# another way
strsum = "0"
sum = 0
for(ch in e)
  if(!is.na(as.numeric(ch))){
    strsum = paste(strsum,ch, sep="")
    sum = sum + as.integer(ch)
  }

print(as.integer(strsum))
print(sum)
#

