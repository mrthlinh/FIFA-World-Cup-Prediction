# -*- coding: utf-8 -*-
"""
Created on Sat May 26 11:06:06 2018

@author: mrthl
"""
#from LE import saveLabelEncoder,loadLabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import numpy as np

def loadData(data, scaler = True, home_team = True,test_size=0.3):
    data_ = data.iloc[:,2:]
    x = data_.iloc[:,:-1]    
    y = data_.iloc[:,-1]
    
    # Label Encoder 'Result'
    encoder = LabelEncoder()    
    y = encoder.fit_transform(y)
    
    if home_team:
        same_ht = x.team_1 == x.home_team
        x.loc[same_ht,'home_team'] = 1
        x.loc[-same_ht,'home_team'] = 0
    else:
        x = x.drop(columns=['home_team'])
    x = x.drop(columns=['team_1','team_2','tournament'])
    
    if scaler:
        x.iloc[:,1:] = StandardScaler().fit_transform(x.iloc[:,1:])
        
    x_train, x_test, y_train, y_test = train_test_split(x,y.squeeze(),test_size=test_size, random_state=85)
    
    return [x,y,x_train, x_test, y_train, y_test]
    