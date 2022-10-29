# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:13:31 2020

 PYR102 week 6 task assignment
 
 @author: KHAN MAHMUDUL HASAN
 
 This task is to apply k-mean clustering to generate new label data
 and to apply logistic regression classification on quantity and 
 categorical independent variables
# 
 Task assignment:
 Library set-up
 
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split # split data into train and test data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
 
 1) read data file (diabetes.csv)
    
"""

diabetes_data=pd.read_csv('diabetes.csv')

"""

 2) select all independent variables(both quantity and categorical)

"""

# select independent data only
diabetes_data1=diabetes_data.iloc[:,0:10]
diabetes_data1  # check for the deletion

"""

 3) perform k-mean clustering with 3 classes

"""

# execute k-mean clustering (number of cluster =3)
# and predict classification labels
pred = KMeans(n_clusters=3).fit_predict(diabetes_data)
print(pred)   # print prediction result

"""

 4) combine class result with data

"""

#
# execute k-mean clustering (number of cluster =3)
# and predict classification labels
pred = KMeans(n_clusters=3).fit_predict(diabetes_data1)
print(pred)   # print prediction result
# assign cluster class to pandas original dataframe
diabetes_data['classid']=pred
# assign cluster class to diabetes_data1 also
diabetes_data1['classid']=pred
#
print(diabetes_data)  # print class for observation
# count number of sample for each cluster
diabetes_data1['classid'].value_counts()
# get mean for each cluster
diabetes_data1[diabetes_data1['classid']==0].mean() # cluster id = 0
diabetes_data1[diabetes_data1['classid']==1].mean() # cluster id = 1
diabetes_data1[diabetes_data1['classid']==2].mean() # cluster id = 2
# visualize stack bar chart
 
clusterinfo = pd.DataFrame()
for i in range(3):
    clusterinfo['cluster' + str(i)] = diabetes_data1[diabetes_data1['classid'] == i].mean()
clusterinfo = clusterinfo.drop('classid')
 
my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 3 Clusters")
my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)
# visualize covariate relationship
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
diabetes_data1["classid"].astype(str)  # change to character numbers
xlist=diabetes_data1.columns.to_list()[0:2]  # get only 2 x variable labels
sns.pairplot(diabetes_data1, hue="classid", vars=xlist)   # plot by SPECIES

# end

"""

 5) Before dividing data, shuffle data

"""



"""

 6) divide data into learning data and test data
#    Learning data size 80%, test data size 20%

"""



"""

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

