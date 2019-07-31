# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:40:59 2019

@author: AN389897
"""

import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pt

dataset = pd.read_csv('Position_Salaries.csv')
del dataset['Position']
X_train = dataset.iloc[:,0:1].values
y_train = dataset.iloc[:,1:2].values

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

regressor = SVR()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)

sc_y.inverse_transform(regressor.predict(sc_X.fit_transform(6.5)))

pt.scatter(X_train, y_train)
pt.plot(X_train, y_pred) 
pt.imshow
