# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:42:08 2018

@author: mrthl
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# User Define Class
from loadData import loadData
from result_report import MyReport
from sklearn.model_selection import train_test_split
#import pickle
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt

# Load data
data_EUR_2012 = pd.read_csv("data/data_full_feature/data_EUR_2012.csv", encoding='utf-8')
data_EUR_2016 = pd.read_csv("data/data_full_feature/data_EUR_2016.csv", encoding='utf-8')

data_WC_2010 = pd.read_csv("data/data_full_feature/data_WC_2010.csv", encoding='utf-8')
data_WC_2014 = pd.read_csv("data/data_full_feature/data_WC_2014.csv", encoding='utf-8')

data_WC_2018 = pd.read_csv("data/data_full_feature/data_WC_2018.csv", encoding='utf-8')

data_all = pd.concat([data_WC_2010,data_EUR_2012,data_WC_2014,data_EUR_2016])
data_classify = data_all.iloc[:,:-1]
data_x, data_y, x_train, x_test, y_train, y_test = loadData(data_classify,home_team=False,scaler = True)

#=====================================================================================
#Synthesize New Data
data_x_new = data_x.apply(lambda row: -row)
data_x_new['h_draw'] = data_x_new['h_draw'].apply(lambda row: -row)
data_x_new['odd_draw'] = data_x_new['odd_draw'].apply(lambda row: -row)

def reverse_result(result):
#    return 0
    # Lose
    if result == 1:
        return 2
    # Win
    elif result == 2:
        return 1
    else:
        return 0
data_y = pd.Series(data_y)
data_y_new = data_y.apply(reverse_result)       

data_x_more = pd.concat([data_x,data_x_new])
data_y_more = pd.concat([data_y,data_y_new])

test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(data_x_more,data_y_more.squeeze(),test_size=test_size, random_state=85)
#=====================================================================================
# Little EDA here

# Feature Importance
RF = RandomForestClassifier(random_state=0,oob_score=True)
#model = RF.fit(data_x,data_y)
model = RF.fit(data_x_more,data_y_more)


def dropcol_importances(rf, X_train, y_train):
#    random_state = 10
    random_state = 999
    rf_ = clone(rf)
    rf_.random_state = random_state
    rf_.fit(X_train, y_train)
#    baseline = rf_.oob_score_
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = random_state
        rf_.fit(X, y_train)
        o = rf_.oob_score_
#        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I

#rf = RandomForestClassifier(oob_score=True)
feature_importance = dropcol_importances(RF, data_x_more, data_y_more)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
#feature_importance = I
#sorted_idx = np.argsort(feature_importance)
pos = np.arange(feature_importance.shape[0]) + .5
#pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance['Importance'], align='center')
#plt.yticks(pos, boston.feature_names[sorted_idx])
plt.yticks(pos, feature_importance.index)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()



feature_importance = model.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
#plt.yticks(pos, boston.feature_names[sorted_idx])
plt.yticks(pos, data_x.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

feature_corr = data_x_more.corr()
plt.matshow(data_x.corr())
cor.style.background_gradient()
cor

#=======================================================================================
# Fit model RandomForest
RF = RandomForestClassifier(random_state=0,oob_score=True)
model = RF.fit(x_train,y_train)
#model = RF.fit(data_x_more,data_y_more)

data_x_18 = data_WC_2018.loc[:,'h_win_diff':]
#RF = RandomForestClassifier(random_state=0)
result = pd.DataFrame()
num_sim = 10000
for state in range(num_sim):
    print(state)
    RF = RandomForestClassifier(random_state =state)
    model = RF.fit(x_train,y_train)
#    model = RF.fit(data_x_more,data_y_more)
#    evaluation = model.score(x_test,y_test)
    
    data_y_18 = model.predict(data_x_18)
    result[state] = data_y_18
    
#    print(evaluation)


result_prob = pd.DataFrame()

result_prob['win'] = result.apply(lambda x: x.value_counts()[2],axis=1)
result_prob['lose'] = result.apply(lambda x: x.value_counts()[1],axis=1)
result_prob['draw'] = result.apply(lambda x: x.value_counts()[0],axis=1)

result_prob['win_prob']  = result_prob.apply(lambda x: 100*x[2] / np.sum(x), axis = 1)
result_prob['lose_prob'] = result_prob.apply(lambda x: 100*x[1] / np.sum(x), axis = 1)
result_prob['draw_prob'] = result_prob.apply(lambda x: 100*x[0] / np.sum(x), axis = 1)

# Report Result
result_report = pd.DataFrame()
result_report['Round'] = data_WC_2018['round']
result_report['Team A'] = data_WC_2018['team_1']
result_report['Team B'] = data_WC_2018['team_2']
result_report['Ti Le Thang (Win Rate)'] = result_prob['win_prob'] 
result_report['Ti Le Hoa (Draw Rate)'] = result_prob['draw_prob'] 
result_report['Ti Le Thua (Lose Rate)'] = result_prob['lose_prob'] 
#row = result.iloc[0,:]
#data_y_18 = model.predict(data_x_18)


#=======================================================================================





