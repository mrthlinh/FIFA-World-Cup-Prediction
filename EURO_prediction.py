# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:57:41 2018

@author: mrthl
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
# User Define Class
from LE import saveLabelEncoder,loadLabelEncoder

# Load Data
data_ = pd.read_csv("data/data_EURO_2016.csv", encoding='utf-8')
data = data_.iloc[:,3:].copy()

# Load Label Encoder
le_result = loadLabelEncoder('LE/result.npy')
data['result'] = le_result.transform(data_['result'])

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

col = list(range(4,18))
col.insert(0,2)
data = data.iloc[:,col]

# Standard Scale
scaler = StandardScaler()
data.iloc[:,1:14] = scaler.fit_transform(data.iloc[:,1:14])

data_x = data.iloc[:,:-1]
data_y = data.iloc[:,-1]

# load the model from disk
#filename = 'save_model/LR.sav'
#filename = 'save_model/RF.sav'
filename = 'save_model/GBT.sav'
#filename = 'save_model/ADA.sav'
loaded_model = pickle.load(open(filename, 'rb'))

y_pred = loaded_model.predict(data_x)
print(classification_report(data_y, y_pred))

result = loaded_model.score(data_x, data_y)
print(result)

y_pred_inv = le_result.inverse_transform(list(y_pred))
data_['prediction'] = y_pred_inv
data_.to_csv('data/data_EURO_2016_result.csv',index = False)
