# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:32:05 2019

@author: Riki jha
"""

import pandas as pd 
import numpy as np

#Breast Cancer detection using knn classifier got 95% accurate model

#dataset is detection of Breast Cancer 
dataset=pd.read_csv('data.csv')

#split x and y
X=dataset.iloc[:,2:32].values
Y=dataset.iloc[:,1].values


#changing Maligant=0 and benign=1
from sklearn.preprocessing import LabelEncoder
labelEncoder_Y=LabelEncoder()
Y=labelEncoder_Y.fit_transform(Y)

#splitting training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,train_size=0.8)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

#Scaling x 
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
knn_classifier=KNeighborsClassifier(metric='minkowski',p=2)

knn_classifier.fit(X_train,y_train)
y_prediction=knn_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_prediction)
