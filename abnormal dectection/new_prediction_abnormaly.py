# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 23:03:57 2018

@author: mrthl
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 10:25:42 2018

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
from sklearn.preprocessing import LabelEncoder
# Load data
data_EUR_2012 = pd.read_csv("data/data_full_feature/abnormaly/data_EUR_2012_abnorm.csv", encoding='utf-8')
data_EUR_2016 = pd.read_csv("data/data_full_feature/abnormaly/data_EUR_2016_abnorm.csv", encoding='utf-8')

data_WC_2010 = pd.read_csv("data/data_full_feature/abnormaly/data_WC_2010_abnorm.csv", encoding='utf-8')
data_WC_2014 = pd.read_csv("data/data_full_feature/abnormaly/data_WC_2014_abnorm.csv", encoding='utf-8')

# For Round 2 of WC 2018
data_WC_2018_1 = pd.read_csv("data/data_full_feature/abnormaly/data_WC_2018_round1_abnorm.csv", encoding='utf-8')
data_WC_2018_2 = pd.read_csv("data/data_full_feature/abnormaly/data_WC_2018_round2.csv", encoding='utf-8')
#data_WC_2018 = pd.read_csv("data/data_full_feature/abnormaly/data_WC_2018_round1_pred.csv", encoding='utf-8')

#data_WC_2018 = pd.read_csv("data/data_full_feature/data_WC_2018_round2.csv", encoding='utf-8')

#data_WC_2018 = pd.read_csv("data/data_full_feature/data_WC_2018.csv", encoding='utf-8')

# Include round 1 of WC2018
data_all = pd.concat([data_WC_2010,data_EUR_2012,data_WC_2014,data_EUR_2016,data_WC_2018_1])
#data_all = pd.concat([data_WC_2010,data_EUR_2012,data_WC_2014,data_EUR_2016])

# Last column is only for Regression
data_classify_abnormal = data_all.iloc[:,:-1]
data_classify_abnormal = data_classify_abnormal.drop(['result'],axis=1)
data_x = data_classify_abnormal.iloc[:,6:-1]
data_y = data_classify_abnormal.iloc[:,-1]

#=====================================================================================
#Synthesize New Data
data_x_new = data_x.iloc[:,1:].apply(lambda row: -row)
data_x_new['h_draw'] = data_x_new['h_draw'].apply(lambda row: -row)
data_x_new['odd_draw'] = data_x_new['odd_draw'].apply(lambda row: -row)
data_x_new['match_type'] = data_x['match_type']
    
data_x_more = pd.concat([data_x,data_x_new])
data_y_more = pd.concat([data_y,data_y])

encoder = LabelEncoder().fit(data_x_more['match_type']) 
np.save("LE/match_type", encoder.classes_) 
data_x_more['match_type'] = encoder.transform(data_x_more['match_type'])

data_WC_2018_2['match_type'] = encoder.transform(data_WC_2018_2['match_type'])
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(data_x_more,data_y_more.squeeze(),test_size=test_size, random_state=85)

#=====================================================================================

RF = RandomForestClassifier(random_state=0,oob_score=True)

data_x_18 = data_WC_2018_2.loc[:,'match_type':]
#RF = RandomForestClassifier(random_state=0)
result = pd.DataFrame()
num_sim = 100000
#num_sim = 10000
feature_importance_list = pd.DataFrame(columns = x_train.columns)
#feature_importance_list.columns = x_train.columns
for state in range(num_sim):
    print(state)
    RF = RandomForestClassifier(random_state =state)
    model = RF.fit(x_train,y_train)
#    model = RF.fit(data_x_more,data_y_more)
#    evaluation = model.score(x_test,y_test)
    importance = model.feature_importances_
    importance = 100.0 * (importance / importance.max())
    feature_importance_list.loc[state,:] = importance
    
    data_y_18 = model.predict(data_x_18)
    result[state] = data_y_18
    
#    print(evaluation)
feature_importance = feature_importance_list.apply(np.mean,axis=0)
feature_importance = feature_importance.sort_values()
feature_importance = 100.0 * (feature_importance / feature_importance.max())

feature_importance.to_csv("temp/feature_importance_abnormaly_round2.csv")

#pos = np.arange(len(feature_importance))+.5
#plt.figure(figsize=(8,6),dpi=80)
#plt.barh(pos, feature_importance.values , align='center')
#plt.yticks(pos,feature_importance.index)
#plt.xlabel('Relative Importance')
#plt.title('Feature Importance')
#plt.show()


result_prob = pd.DataFrame()

result_prob['1'] = result.apply(lambda x: x.value_counts().get(1,0),axis=1)
result_prob['0'] = result.apply(lambda x: x.value_counts().get(0,0),axis=1)

result_prob['abnormal_prob'] = result_prob.apply(lambda x: 100*x[0] / np.sum(x), axis = 1)
# Report Result

result_report = pd.DataFrame()
result_report['Round'] = data_WC_2018_2['id']
result_report['Team A'] = data_WC_2018_2['team_1']
result_report['Team B'] = data_WC_2018_2['team_2']
result_report['Abnormal'] = result_prob['abnormal_prob'] 
result_report['odd_diff_win'] = data_WC_2018_2['odd_diff_win']
result_report['odd_draw'] = data_WC_2018_2['odd_draw']


result_report.to_csv("temp/abnormal_round2.csv",index=False)

