# -*- coding: utf-8 -*-
"""
Created on Thu May 24 21:51:31 2018

@author: mrthl
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from LE import saveLabelEncoder,loadLabelEncoder
from sklearn.preprocessing import StandardScaler
# Load Data
data = pd.read_csv("data/data_odd_2005_regression_syn.csv", encoding='utf-8')
data = data.iloc[:,2:].copy()
# Load Label Encoder
le_result = loadLabelEncoder('LE/result.npy')
data['result'] = le_result.transform(data['result'])

le_tour = loadLabelEncoder('LE/tournament.npy')
data['tournament'] = le_tour.transform(data['tournament'])

le_country = loadLabelEncoder('LE/country.npy')
data['team_1'] = le_country.transform(data['team_1'])
data['team_2'] = le_country.transform(data['team_2'])
data['home_team'] = le_country.transform(data['home_team'])

# Add HOME team
same_ht = data.team_1 == data.home_team
data.loc[same_ht,'home_team'] = 1
data.loc[-same_ht,'home_team'] = 0

col = list(range(4,19))
col.insert(0,2)
data = data.iloc[:,col]

# Drop result column
data = data.drop(columns=['result'])

#
#scaler = StandardScaler()
#data.iloc[:,1:14] = scaler.fit_transform(data.iloc[:,1:14])

# Split data training and testing
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1:].squeeze(),test_size=0.3, random_state=85)
data_x = data.iloc[:,:-1]
data_y = data.iloc[:,-1].squeeze()


# Fit regression model
params = {'n_estimators': 600, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(x_train, y_train)
mse = mean_squared_error(y_test, clf.predict(x_test))
print("MSE: %.4f" % mse)

# Make predictions using the testing set
y_pred = clf.predict(x_test)

# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(x_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
#plt.yticks(pos, boston.feature_names[sorted_idx])
plt.yticks(pos, x_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()



from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge

# Fit regression model
#train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=3,
                   param_grid={"C": [1e0, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 3)},verbose = True)

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=3,
                  param_grid={"alpha": [1e0, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 3)},verbose = True)


svr.fit(x_train, y_train)

kr.fit(x_train, y_train)

#sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
#print("Support vector ratio: %.3f" % sv_ratio)

y_svr = svr.predict(x_test)

y_kr = kr.predict(x_test)

mse = mean_squared_error(y_test, svr.predict(x_test))
print("MSE: %.4f" % mse)

mse = mean_squared_error(y_test, kr.predict(x_test))
print("MSE: %.4f" % mse)

# Adaboost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),learning_rate = 0.01,n_estimators=200)

ada_reg.fit(x_train, y_train)
mse = mean_squared_error(y_test, ada_reg.predict(x_test))
print("MSE: %.4f" % mse)

# Isotonic
from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(hidden_layer_sizes = [1000,500,250], max_iter=1000, tol = 1e-5, 
                  verbose = True,early_stopping=True)
nn.fit(x_train, y_train)
mse_train = mean_squared_error(y_train, nn.predict(x_train))
print("MSE training: %.4f" % mse_train)

mse_test = mean_squared_error(y_test, nn.predict(x_test))
print("MSE test: %.4f" % mse_test)