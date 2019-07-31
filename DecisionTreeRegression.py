# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 00:41:56 2019

@author: AN389897
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pt

dataset = pd.read_csv('Position_Salaries.csv')
del dataset['Position']
X_train = dataset.iloc[:,0:1].values
y_train = dataset.iloc[:,1:2].values

#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#y_train = sc_y.fit_transform(y_train)

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)

regressor.predict(6.5)
X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
pt.scatter(X_train, y_train)
pt.plot(X_grid, regressor.predict(X_grid)) 
pt.show
