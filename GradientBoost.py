# -*- coding: utf-8 -*-
"""
Created on Sat May 26 23:32:54 2018

@author: mrthl
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
# User Define Class
from loadData import loadData
from result_report import MyReport
import pickle

# Load data
data = pd.read_csv("data/data_odd_2005.csv", encoding='utf-8')
data_x, data_y, x_train, x_test, y_train, y_test = loadData(data)
list_data = [data_x, data_y, x_train, x_test, y_train, y_test]

# Fit GradientBoosting Tree with GridSearch CV:
GBT = GradientBoostingClassifier(random_state=0, verbose = True)
grid_GBT = [{'max_depth': [3,5,7], 'n_estimators': [100,1000,2000]}]
clf_GBT = GridSearchCV(estimator=GBT,param_grid = grid_GBT,scoring='f1_micro',
                   cv = 3,n_jobs=1,verbose=True)

# Fit cross-validation and report result
modelCV_GBT = MyReport(model = clf_GBT, model_Name = 'Gradient Boosting Tree', list_data = list_data,tune = True)

# Save the best model
filename = 'save_model/GBT.sav'
pickle.dump(modelCV_GBT, open(filename, 'wb'))