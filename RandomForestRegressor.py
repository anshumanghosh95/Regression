# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:58:53 2019

@author: AN389897
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as pt

dataset = pd.read_csv('Position_Salaries.csv')
del dataset['Position']
X_train = dataset.iloc[:,0:1].values
y_train = dataset.iloc[:,1:2].values

regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)

regressor.predict(6.5)

X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
pt.scatter(X_train, y_train)
pt.plot(X_grid, regressor.predict(X_grid)) 
pt.show
