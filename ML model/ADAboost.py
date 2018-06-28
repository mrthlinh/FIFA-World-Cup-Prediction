# -*- coding: utf-8 -*-
"""
Created on Sat May 26 23:38:34 2018

@author: mrthl
"""
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
ADAboost = AdaBoostClassifier(algorithm="SAMME",learning_rate=1)

DT_3 = DecisionTreeClassifier(max_depth=3)
DT_5 = DecisionTreeClassifier(max_depth=5)
grid_ADA = [{'base_estimator': [DT_3,DT_5], 'n_estimators': [100,1000,2000,3000]}]
 
clf_ADA= GridSearchCV(estimator=ADAboost,param_grid = grid_ADA,scoring='f1_micro',
                   cv = 3,n_jobs=1,verbose=True)

# Fit cross-validation and report result
modelCV_ADA = MyReport(model = clf_ADA, model_Name = 'ADA Boosting Tree', list_data = list_data,tune = True)

# Save the best model
filename = 'save_model/ADA.sav'
pickle.dump(modelCV_ADA, open(filename, 'wb'))