# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 11:28:15 2019

@author: Harsh Anand
"""

import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns


#load dataset
data = pd.read_csv("creditcard.csv")

#explore dataset
print(data.columns)

print(data.describe())

#take 10% of data using frac and random_state is used for same type of data
data = data.sample(frac = 0.1,random_state = 1)
print(data.shape)

#plot histogram of each parameter
data.hist(figsize = (20,20))

#determine number of fraud cases in data
fraud = data[data['Class']==1]
valid = data[data['Class']==0] 
print("fraud cases : {}".format(len(fraud)))
print("valid cases : {}".format(len(valid)))

#calculate % of fraud cases to valid cases
#float is used to obtain the result in float
outlier = len(fraud) / float(len(valid))
print("% of outlier for fraud cases to valid cases",outlier)

#coorelation matrix --> to see is there any strong coorelation b/w different variables in dataset
#and it also yo show which feature are important for overall classification and help to choose linear methods
corrmatrix = data.corr()

fig = plt.figure(figsize = (12,9)) #to give the shape of graph

''' seaborn is used to data visualisation and  It provides a high-level interface for drawing attractive and informative statistical graphics
and we use heatmap graph in this and VMAX-->Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments  '''

sns.heatmap(corrmatrix, vmax=.8, square=True)
plt.show()

#get all the columns from dataframe
columns = data.columns.tolist()

features = data.iloc[:,:-1] #taking all coulmn except class
labels = data.iloc[:,-1] #taking column class

#******************************************************************************
'''
-- Unsupervised Outlier Detection--
Now that we have processed our data, we can begin deploying our machine learning algorithms. We will use the following techniques:

--Local Outlier Factor (LOF)--
The anomaly score of each sample is called Local Outlier Factor. It measures the local deviation of density of a given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood.

--Isolation Forest Algorithm--
The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.
This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.
'''

from sklearn.metrics import classification_report, accuracy_score #metrices are used for how successful we are in outlier detection & accuracy score
from sklearn.ensemble import IsolationForest 
from sklearn.neighbors import LocalOutlierFactor #these are two common anamoly detection package and these are based on neighbour items
# define outlier detection tools to be compared and stored in dictionary
classifier = {
        "Isolation Forest":IsolationForest(max_samples=len(features),
                                           contamination = outlier,
                                           random_state = 1),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20,
                                                   contamination = outlier)
        }

#fit the model
num_outlier = len(fraud)

#enumerate gives index & value 
#classifier is dictionary so we pass (clf_name(key)--> as model name, clf(value)--> as model fitting)
for i, (clf_name, clf) in enumerate(classifier.items()):
    #fit data and tag outlier
    if clf_name == "Local Outlier Factor":
        labels_pred = clf.fit_predict(features)
        #scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(features)
        #scores_pred = clf.decision_function(features)
        labels_pred = clf.predict(features)
        
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    labels_pred[labels_pred == 1] = 0
    labels_pred[labels_pred == -1] = 1
    
    number_of_errors = (labels_pred != labels).sum() 
    #count the number when labels & labels_pred does not match
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, number_of_errors))
    print(accuracy_score(labels, labels_pred))
    print(classification_report(labels, labels_pred))
        