# -*- coding: utf-8 -*-
"""
Created on Sat May 26 23:45:38 2018

@author: mrthl
"""
import pandas as pd
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import GridSearchCV
# User Define Class
from loadData import loadData
from result_report import MyReport
import pickle

# Load data
data = pd.read_csv("data/data_odd_2005.csv", encoding='utf-8')
data_x, data_y, x_train, x_test, y_train, y_test = loadData(data)
list_data = [data_x, data_y, x_train, x_test, y_train, y_test]

# Fit GradientBoosting Tree with GridSearch CV:
layer = (1000,500)
model_NN = MLPClassifier(hidden_layer_sizes = layer, max_iter = 1000, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-10, random_state=1,
                    learning_rate_init=.1)

# Fit cross-validation and report result
modelCV_NN = MyReport(model = model_NN, model_Name = 'Neural Network', list_data = list_data,tune = False)

# Save the best model
filename = 'save_model/NN.sav'
pickle.dump(modelCV_NN, open(filename, 'wb'))
