# -*- coding: utf-8 -*-
"""
Created on Sat May 26 22:58:15 2018

@author: mrthl
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# User Define Class
from loadData import loadData
from result_report import MyReport
import pickle

# Load data
data = pd.read_csv("data/data_odd_2005.csv", encoding='utf-8')
data_x, data_y, x_train, x_test, y_train, y_test = loadData(data)
list_data = [data_x, data_y, x_train, x_test, y_train, y_test]

# Fit Random Forest with GridSearch CV:
RF = RandomForestClassifier(random_state=0)
grid_RF= [{'n_estimators': [5,10,15]}] 
clf_RF = GridSearchCV(estimator = RF,param_grid = grid_RF,scoring='f1_micro',cv = 3,n_jobs=1,verbose = True)

# Fit cross-validation and report result
modelCV_RF = MyReport(model = clf_RF, model_Name = 'Random Forest', list_data = list_data,tune = True)

# Save the best model
filename = 'save_model/RF.sav'
pickle.dump(modelCV_RF, open(filename, 'wb'))