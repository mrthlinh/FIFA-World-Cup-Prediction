# -*- coding: utf-8 -*-
"""
Created on Tue May  8 09:41:27 2018

@author: mrthl
"""
# Load library
#import csv
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# User Define Class
from test_LE import saveLabelEncoder,loadLabelEncoder
from result_plot import plot_confusion_matrix,plot_ROC_curve
# Load Data
data_ = pd.read_csv("data/data_odd_2005.csv", encoding='utf-8')
data = data_.iloc[:,2:].copy()

# Label Encoder
le = preprocessing.LabelEncoder()
le_tour = saveLabelEncoder(data['tournament'],'LE/tournament.npy')
data['tournament'] = le_tour.transform(data['tournament'])
le_result = saveLabelEncoder(data['result'],'LE/result.npy')
data['result'] = le_result.transform(data['result'])
x = np.concatenate((data['team_1'],data['team_2']), axis=0)
le_country = saveLabelEncoder(x,'LE/country.npy')
data['team_1'] = le_country.transform(data['team_1'])
data['team_2'] = le_country.transform(data['team_2'])
data['home_team'] = le_country.transform(data['home_team'])

# Standard Scale
data_test = data.iloc[:,4:].copy()
data_test[data_test.columns[:-1]] = MinMaxScaler().fit_transform(data_test[data_test.columns[:-1]])
data = data_test.copy()



# Split data training and testing
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1:].squeeze(),test_size=0.3, random_state=85)



#==========================================================
# Fit a simple LR
model_LR = LogisticRegression(multi_class='ovr',penalty='l1', C=0.1).fit(x_train,y_train)
model_LR.score(x_train,y_train)
y_pred_LR = model_LR.predict(x_test)
print(classification_report(y_test, y_pred_LR))
accuracy_score(y_test, y_pred_LR)
accuracy_score(y_train, model_LR.predict(x_train))
#==========================================================
# Fit a LR with GridSearchCV
LR = LogisticRegression(multi_class='ovr',penalty='l1')
grid_LR = [{'C': np.logspace(-6, 0, 10)}]
clf_LR = GridSearchCV(estimator = LR,param_grid = grid_LR,scoring='f1_micro',
                   cv = 5,n_jobs=-1)
clf_LR.fit(x_train,y_train)
clf_LR.best_score_                                
modelCV_LR = clf_LR.best_estimator_           
y_predCV_LR = modelCV_LR.predict(x_test)
print(classification_report(y_test, y_predCV_LR))
accuracy_score(y_test, y_predCV_LR)

for params, mean_score, scores in clf_LR.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

#==========================================================
# Fit a SVM with GridSearchCV
SVM = SVC()
grid_SVM = [{'kernel': ['rbf'], 'C': np.logspace(-6, 0, 5),'degree': [3,4,5]}]
 
clf_SVM = GridSearchCV(estimator=SVM,param_grid = grid_SVM,scoring='accuracy',
                   cv = 5,n_jobs=-1)
clf_SVM.fit(x_train,y_train)
clf_SVM.best_score_                                 
model_SVM = clf_SVM.best_estimator_           
y_pred_SVM = model_SVM.predict(x_test)
print(classification_report(y_test, y_pred_SVM))
accuracy_score(y_test, y_pred_SVM)

for params, mean_score, scores in clf_SVM.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
#==========================================================
# Fit a DecisionTree

clf_DT = tree.DecisionTreeClassifier()
model_DT = clf_DT.fit(x_train,y_train)
    
#model_SVM = clf_SVM.best_estimator_           
y_pred_DT = model_DT.predict(x_test)
print(classification_report(y_test, y_pred_DT))
accuracy_score(y_test, y_pred_DT)

#==========================================================

# Fit Random Forest

model_RF = RandomForestClassifier(max_depth=10, random_state=0).fit(x_train,y_train)

#print(model_RF.feature_importances_)

y_pred_RF = model_RF.predict(x_test)
print(classification_report(y_test, y_pred_RF))
accuracy_score(y_test, y_pred_RF)
 
#==========================================================

# Fit a shallow NN

#model_NN = MLPClassifier(hidden_layer_sizes = (500,400,300,100,50), max_iter = 1000, alpha=1e-4,
#                    solver='adam', verbose=10, tol=1e-4, random_state=1,
#                    learning_rate_init=.1).fit(x_train,y_train)


model_NN = MLPClassifier(hidden_layer_sizes = (500,300,50), max_iter = 1000, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-10, random_state=1,
                    learning_rate_init=.1).fit(x_train,y_train)

y_pred_NN = model_NN.predict(x_test)
print(classification_report(y_test, y_pred_NN))
accuracy_score(y_test, y_pred_NN)

print("Training set score: %f" % model_NN.score(x_train, y_train))
print("Test set score: %f" % model_NN.score(x_test, y_test))

#==========================================================
bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

model_ADA_1 = bdt_real.fit(x_train,y_train)

model_ADA_2 = bdt_discrete.fit(x_train,y_train)

y_pred_ADA_1  = model_ADA_1.predict(x_test)
print(classification_report(y_test, y_pred_ADA_1))
accuracy_score(y_test, y_pred_ADA_1)

y_pred_ADA_2  = model_ADA_2.predict(x_test)
print(classification_report(y_test, y_pred_ADA_2))
accuracy_score(y_test, y_pred_ADA_2)

#==========================================================

# Plot non-normalized confusion matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')

# Classification Report
print(classification_report(y_test.squeeze(), y_pred))

# ROC curve
y_score = model_LR.decision_function(x_test)
plot_ROC_curve(y_test,y_score)

#Compare many classifiers
#http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py