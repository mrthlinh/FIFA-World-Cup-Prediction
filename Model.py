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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

import pickle

# User Define Class
from LE import saveLabelEncoder,loadLabelEncoder
from result_plot import plot_confusion_matrix,plot_ROC_curve

# Load Data
data_ = pd.read_csv("data/data_odd_2005.csv", encoding='utf-8')
data = data_.iloc[:,2:].copy()

# Label Encoder (only do 1 time)
#le = preprocessing.LabelEncoder()
#le_tour = saveLabelEncoder(data['tournament'],'LE/tournament.npy')
#data['tournament'] = le_tour.transform(data['tournament'])
#le_result = saveLabelEncoder(data['result'],'LE/result.npy')
#data['result'] = le_result.transform(data['result'])
#x = np.concatenate((data['team_1'],data['team_2']), axis=0)
#le_country = saveLabelEncoder(x,'LE/country.npy')
#data['team_1'] = le_country.transform(data['team_1'])
#data['team_2'] = le_country.transform(data['team_2'])
#data['home_team'] = le_country.transform(data['home_team'])

# Load Label Encoder
le_result = loadLabelEncoder('LE/result.npy')
data['result'] = le_result.transform(data_['result'])

le_tour = loadLabelEncoder('LE/tournament.npy')
data['tournament'] = le_tour.transform(data['tournament'])

le_country = loadLabelEncoder('LE/country.npy')
data['team_1'] = le_country.transform(data['team_1'])
data['team_2'] = le_country.transform(data['team_2'])
data['home_team'] = le_country.transform(data['home_team'])

# Standard Scale
#data_test = data.iloc[:,4:].copy()
#data_test[data_test.columns[:-1]] = MinMaxScaler().fit_transform(data_test[data_test.columns[:-1]])
#data = data_test.copy()


#scaler = StandardScaler()
#scaler.fit(x_train.iloc[:,1:])
#
#x_train.iloc[:,1:] = scaler.transform(x_train.iloc[:,1:])
#x_test.iloc[:,1:]= scaler.transform(x_test.iloc[:,1:])

# Add HOME team
same_ht = data.team_1 == data.home_team
data.loc[same_ht,'home_team'] = 1
data.loc[-same_ht,'home_team'] = 0

col = list(range(4,18))
col.insert(0,2)
data = data.iloc[:,col]

scaler = StandardScaler()
data.iloc[:,1:14] = scaler.fit_transform(data.iloc[:,1:14])

# Split data training and testing
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1:].squeeze(),test_size=0.3, random_state=85)
data_x = data.iloc[:,:-1]
data_y = data.iloc[:,-1].squeeze()

# Scale
#scaler = StandardScaler()
#scaler.fit(x_train.iloc[:,1:])
#
#x_train.iloc[:,1:] = scaler.transform(x_train.iloc[:,1:])
#x_test.iloc[:,1:]= scaler.transform(x_test.iloc[:,1:])


# Save filte train and test
x_train.to_csv('data/train_x.csv',index = False)
x_test.to_csv('data/test_x.csv',index = False)
y_train.to_csv('data/train_y.csv',index = False)
y_test.to_csv('data/test_y.csv',index = False)

#x_train_ = pd.read_csv("data/train_x.csv", encoding='utf-8')
#==========================================================
# Fit a simple LR
model_LR = LogisticRegression(multi_class='ovr',penalty='l1', C=0.1).fit(x_train,y_train)
model_LR.score(x_train,y_train)
y_pred_LR = model_LR.predict(x_test)
print(classification_report(y_test, y_pred_LR))
accuracy_score(y_test, y_pred_LR)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_LR)
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Logistic Regression Confusion matrix, without normalization')

# ROC curve
y_score = model_LR.decision_function(x_test)
plot_ROC_curve(y_test,y_score,title='Logistic Regression ROC curve',class_names = class_names)
#==========================================================
# Fit a LR with GridSearchCV
#LR = LogisticRegression(multi_class='ovr',penalty='l1' )
LR = LogisticRegression(multi_class='multinomial',penalty='l2',solver='lbfgs')
grid_LR = [{'C': np.logspace(-6, 0, 10)}]
#grid_LR = [{'C': np.logspace(-6, 0, 10), 'penalty' : ['l1','l2']}]
#clf_LR = GridSearchCV(estimator = LR,param_grid = grid_LR,scoring='f1_micro',
#                   cv = 5,n_jobs=-1,verbose = True)
clf_LR = GridSearchCV(estimator = LR,param_grid = grid_LR,scoring='f1_micro',
                   cv = 5,verbose = True)
clf_LR.fit(x_train,y_train)
clf_LR.best_score_                                
modelCV_LR = clf_LR.best_estimator_           
y_predCV_LR = modelCV_LR.predict(x_test)
print(classification_report(y_test, y_predCV_LR))
accuracy_score(y_test, y_predCV_LR)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_predCV_LR)
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Logistic Regression Confusion matrix, without normalization')

#plot_confusion_matrix(cnf_matrix, classes=class_names,normalize = True,title='Normalized Confusion matrix')

# ROC curve
y_score = modelCV_LR.decision_function(x_test)
#plot_ROC_curve(y_test,y_score,class_names,title='Logistic Regression ROC curve' )
plot_ROC_curve(y_test,y_score,title='Logistic Regression ROC curve',class_names = class_names)

for params, mean_score, scores in clf_LR.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

# 10-fold-test error
scores = cross_val_score(modelCV_LR, data_x, data_y, cv=10)
print(scores)
print(np.mean(scores))

# Save the best model
filename = 'save_model/LR.sav'
pickle.dump(modelCV_LR, open(filename, 'wb'))
    
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)
#==========================================================
# Fit a simple SVM 
#SVM_ = SVC(C=0.001,kernel = 'linear')
SVM_ = SVC(kernel = 'rbf')
model_SVM_ = SVM_.fit(x_train,y_train)
y_pred_SVM_ = model_SVM_.predict(x_test)
print(classification_report(y_test, y_pred_SVM_))
accuracy_score(y_test, y_pred_SVM_)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_SVM_)
plot_confusion_matrix(cnf_matrix, classes=class_names,title='SVM Confusion matrix, without normalization')

y_score = model_SVM_.decision_function(x_test)
#plot_ROC_curve(y_test,y_score,class_names,title='Logistic Regression ROC curve' )
plot_ROC_curve(y_test,y_score,title='SVM ROC curve',class_names = class_names)

#==========================================================
# Fit a simple SVM  linear
from sklearn.svm import LinearSVC
model_LinearSVC = LinearSVC(random_state=0,verbose = True,max_iter=10000).fit(x_train,y_train)
y_pred_SVM_ = model_LinearSVC.predict(x_test)
print(classification_report(y_test, model_LinearSVC))
accuracy_score(y_test, y_pred_SVM_)
#==========================================================

# Fit a SVM with GridSearchCV
SVM = SVC(verbose = True)
grid_SVM = [{'kernel': ['rbf','linear'], 'C': np.logspace(-6, 0, 3)}]
 
clf_SVM = GridSearchCV(estimator=SVM,param_grid = grid_SVM,scoring='f1_macro',
                   cv = 5,n_jobs=-1,verbose = True)
clf_SVM.fit(x_train,y_train)
clf_SVM.best_score_                                 
model_SVM = clf_SVM.best_estimator_           
y_pred_SVM = model_SVM.predict(x_test)
print(classification_report(y_test, y_pred_SVM))
accuracy_score(y_test, y_pred_SVM)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_SVM)
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='SVM Confusion matrix, without normalization')

plot_confusion_matrix(cnf_matrix, classes=class_names,normalize = True,title='Normalized Confusion matrix')

y_score = model_SVM.decision_function(x_test)
#plot_ROC_curve(y_test,y_score,class_names,title='Logistic Regression ROC curve' )
plot_ROC_curve(y_test,y_score,title='SVM ROC curve',class_names = class_names)


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
model_RF  = RandomForestClassifier(n_estimators=2000,random_state=0, n_jobs = -1, verbose = True).fit(x_train,y_train)
y_pred_RF = model_RF.predict(x_test)
print(classification_report(y_test, y_pred_RF))
accuracy_score(y_test, y_pred_RF)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_RF)
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Random Forest Confusion matrix, without normalization')

#==========================================================
# Fit Random Forest with GridSearch CV:
RF = RandomForestClassifier(random_state=0)
grid_RF= [{'n_estimators': [5,10,15]}] 
clf_RF = GridSearchCV(estimator = RF,param_grid = grid_RF,scoring='f1_micro',cv = 3,n_jobs=1,verbose = True)
clf_RF.fit(x_train,y_train)

clf_RF.best_score_                                 
model_RF = clf_RF.best_estimator_           
y_pred_RF = model_RF.predict(x_test)
print(classification_report(y_test, y_pred_RF))
accuracy_score(y_test, y_pred_RF)

#print(model_RF.feature_importances_)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_RF)
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Random Forest Confusion matrix, without normalization')

plot_confusion_matrix(cnf_matrix, classes=class_names,normalize = True,title='Normalized Confusion matrix')

# 10-fold-test error
scores = cross_val_score(model_RF, data_x, data_y, cv=10)
print(scores)
print(np.mean(scores))

# Save the best model
filename = 'save_model/RF.sav'
pickle.dump(model_RF, open(filename, 'wb'))

#==========================================================
# GradientBoostingClassifier: need CV herer
GBT = GradientBoostingClassifier(random_state=0, verbose = True)
grid_GBT = [{'max_depth': [3,5,7], 'n_estimators': [100,1000,2000]}]
clf_GBT = GridSearchCV(estimator=GBT,param_grid = grid_GBT,scoring='f1_micro',
                   cv = 3,n_jobs=1,verbose=True)

clf_GBT.fit(x_train,y_train)
clf_GBT.best_score_                                 
model_GBT = clf_GBT.best_estimator_   

#model_GB  = GradientBoostingClassifier(n_estimators=1000,max_depth = 7,random_state=0, verbose = True).fit(x_train,y_train)
y_pred_GBT = model_GBT.predict(x_test)
print(classification_report(y_test, y_pred_GBT))
accuracy_score(y_test, y_pred_GBT)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_GBT)
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Gradient Boosting DT Confusion matrix, without normalization')

# ROC curve
y_score = model_GBT.decision_function(x_test)
plot_ROC_curve(y_test,y_score,title='Gradient Boosting DT ROC curve',class_names = class_names)

# 10-fold-test error
scores = cross_val_score(model_GBT, data_x, data_y, cv=10)
print(scores)
print(np.mean(scores))

# Save the best model
filename = 'save_model/GBT.sav'
pickle.dump(model_GBT, open(filename, 'wb'))

#==========================================================
# Fit a ADAboost Tree: seem promising, need CV here
ADAboost = AdaBoostClassifier(algorithm="SAMME",learning_rate=1)

DT_3 = DecisionTreeClassifier(max_depth=3)
DT_5 = DecisionTreeClassifier(max_depth=5)
grid_ADA = [{'base_estimator': [DT_3,DT_5], 'n_estimators': [100,1000,2000,3000]}]
 
clf_ADA= GridSearchCV(estimator=ADAboost,param_grid = grid_ADA,scoring='accuracy',
                   cv = 3,n_jobs=1,verbose=True)
clf_ADA.fit(x_train,y_train)

clf_ADA.best_score_                                 
model_ADA = clf_ADA.best_estimator_       

y_pred_ADA  = model_ADA.predict(x_test)
print(classification_report(y_test, y_pred_ADA))
accuracy_score(y_test, y_pred_ADA)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_ADA)
plot_confusion_matrix(cnf_matrix, classes=class_names,title='ADAboost DT Confusion matrix, without normalization')

# ROC curve
y_score = model_ADA.decision_function(x_test)
plot_ROC_curve(y_test,y_score,title='AdaBoost DT ROC curve',class_names = class_names)

# 10-fold-test error
scores = cross_val_score(model_ADA, data_x, data_y, cv=10)
print(scores)
print(np.mean(scores))

# Save the best model
filename = 'save_model/ADA.sav'
pickle.dump(model_ADA, open(filename, 'wb'))
#==========================================================
from sklearn.naive_bayes import MultinomialNB
model_NB = MultinomialNB().fit(x_train, y_train)
y_pred_NB = model_NB.predict(x_test)
print(classification_report(y_test, y_pred_NB))
accuracy_score(y_test, y_pred_NB)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_NB)
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Naive Bayes Confusion matrix, without normalization')

#==========================================================
# Fit a shallow NN
#model_NN = MLPClassifier(hidden_layer_sizes = (500,400,300,100,50), max_iter = 1000, alpha=1e-4,
#                    solver='adam', verbose=10, tol=1e-4, random_state=1,
#                    learning_rate_init=.1).fit(x_train,y_train)

# Standard Scale
# Add Home Team

#data = orig_data.copy()
#orig_data = data.copy()
#same_ht = data.team_1 == data.home_team
#data.loc[same_ht,'home_team'] = 1
#data.loc[-same_ht,'home_team'] = 0
#
#col = list(range(4,18))
#col.insert(0,2)
#data_test = data.iloc[:,col].copy()

#data_test = data.iloc[:,4:].copy()
#data_test[data_test.columns[:-1]] = MinMaxScaler().fit_transform(data_test[data_test.columns[:-1]])
#data = data_test.copy()

# Split data training and testing
#x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1:].squeeze(),test_size=0.3, random_state=85)

# Scale
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(x_train.iloc[:,1:])
#
#x_train.iloc[:,1:] = scaler.transform(x_train.iloc[:,1:])
#x_test.iloc[:,1:]= scaler.transform(x_test.iloc[:,1:])
#
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

#from sklearn.preprocessing import LabelBinarizer
#lb = LabelBinarizer().fit(y_train)
#lb.classes_
#lb.transform([0,1,2])
#
#y_train_lb = lb.transform(y_train)
#y_test_lb = lb.transform(y_test)
#
#y_train_draw = y_train_lb[:,0]
#y_test_draw = y_test_lb[:,0]

# only train on Draw
#layer = (1000,500,250,100)
layer = (500,250,100,50,25)
alpha = 0.1
model_NN = MLPClassifier(hidden_layer_sizes = layer, max_iter = 1000, alpha=alpha, 
                         learning_rate = 'adaptive', activation = 'tanh',early_stopping =False,
                        solver='adam', verbose=True, tol=1e-6, random_state=1,
                        learning_rate_init=.001).fit(x_train,y_train)

y_pred_NN = model_NN.predict(x_test)
print(classification_report(y_test, y_pred_NN))
accuracy_score(y_test, y_pred_NN)

print("Training set score: %f" % model_NN.score(x_train, y_train))
print("Test set score: %f" % model_NN.score(x_test, y_test))

# k-fold Cross validation error => better estimation of error

data_x = data.iloc[:,:-1]
data_y = data.iloc[:,-1].squeeze()

layer = (1000,500)
alpha = 0.1
model_NN = MLPClassifier(hidden_layer_sizes = layer, max_iter = 1000, alpha=alpha, 
                         learning_rate = 'adaptive', activation = 'tanh',early_stopping =True,
                        solver='adam', verbose=True, tol=1e-6, random_state=1,
                        learning_rate_init=.001)

scores = cross_val_score(model_NN, data_x, data_y, cv=10)
print(scores)
print(np.mean(scores))
#========================
model_NN = MLPClassifier(hidden_layer_sizes = (13,10), max_iter = 1000, alpha=1e-4,
                    solver='adam', verbose=True, tol=1e-10, random_state=1,
                    learning_rate_init=.1).fit(x_train,y_train)

#model_NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(13, 10), random_state=1).fit(x_train,y_train)

y_pred_NN = model_NN.predict(x_test)
print(classification_report(y_test, y_pred_NN))
accuracy_score(y_test, y_pred_NN)

print("Training set score: %f" % model_NN.score(x_train, y_train))
print("Test set score: %f" % model_NN.score(x_test, y_test))

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_NN)
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Random Forest Confusion matrix, without normalization')

plot_confusion_matrix(cnf_matrix, classes=class_names,normalize = True,title='Normalized Confusion matrix')

#y_score = model_NN.decision_function(x_test)
##plot_ROC_curve(y_test,y_score,class_names,title='Logistic Regression ROC curve' )
#plot_ROC_curve(y_test,y_score,title='Random Forest ROC curve',class_names = class_names)
#==========================================================
#Fit with xgboost

model_xgb = XGBClassifier(n_estimators = 100,max_depth=6,silent = False)
model_xgb.fit(x_train, y_train)
# make predictions for test data
y_pred = model_xgb.predict(x_test)
#predictions = [round(value) for value in y_pred]
# evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

print("Training set score: %f" % model_xgb.score(x_train, y_train))
print("Test set score: %f" % model_xgb.score(x_test, y_test))

#Compare many classifiers
#http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

#==================================================
#fit KNN
filename = 'save_model/PCA.sav'
pca_loaded_model = pickle.load(open(filename, 'rb'))
x_train_pc = pca_loaded_model.transform(x_train)
x_test_pc = pca_loaded_model.transform(x_test)

neigh = KNeighborsClassifier(n_neighbors=9)
neigh.fit(x_train_pc,y_train)
y_pred_KNN = neigh.predict(x_test_pc)
print(classification_report(y_test, y_pred_KNN))
accuracy_score(y_test, y_pred_KNN)

# Plot Confusion Matrix
class_names = le_result.classes_
cnf_matrix = confusion_matrix(y_test, y_pred_KNN)
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='KNN Confusion matrix, without normalization')

