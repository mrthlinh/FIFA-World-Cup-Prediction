# -*- coding: utf-8 -*-
"""
Created on Sat May 26 12:17:08 2018

@author: mrthl
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# User Define Class
from loadData import loadData
from result_report import MyReport
import pickle

# Load data
data = pd.read_csv("data/data_odd_2005.csv", encoding='utf-8')
data_x, data_y, x_train, x_test, y_train, y_test = loadData(data)
list_data = [data_x, data_y, x_train, x_test, y_train, y_test]

# Define Logistic Regression and Hyper-parameter Grid
LR = LogisticRegression(multi_class='multinomial',penalty='l2',solver='lbfgs')
grid_LR = [{'C': np.logspace(-6, 0, 10)}]
clf_LR = GridSearchCV(estimator = LR,param_grid = grid_LR,scoring='f1_micro',
                   cv = 5,verbose = True)

# Fit cross-validation and report result
modelCV_LR = MyReport(model = clf_LR, model_Name = 'Logistic Regression', list_data = list_data,tune = True)

# Save the best model
filename = 'save_model/LR.sav'
pickle.dump(modelCV_LR, open(filename, 'wb'))
