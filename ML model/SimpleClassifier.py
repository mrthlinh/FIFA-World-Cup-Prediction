# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:49:29 2018

@author: mrthl
"""
import pandas as pd
#from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import GridSearchCV
# User Define Class
from loadData import loadData
from result_report import MyReport
import pickle

# Load data
data = pd.read_csv("data/data_odd_2005.csv", encoding='utf-8')
data_x, data_y, x_train, x_test, y_train, y_test = loadData(data,scaler=False)
list_data = [data_x, data_y, x_train, x_test, y_train, y_test]

def infer_result(row):
    thres_w = 1.5
#    thres_d = 0
#    thres_d2 = 0
    diff_w = row['avg_odds_win_1'] - row['avg_odds_win_2']
#    diff_d1 = row['avg_odds_draw'] - row['avg_odds_win_1'] 
#    diff_d2 = row['avg_odds_draw'] - row['avg_odds_win_2'] 
#    if (diff_w >= thres_w ):
#        if (diff_d2 <= thres_d):
#            return 'lose'
#        else:
#            return 'draw'
#    elif (diff_w =< -thres_w ):
#        if (diff_d1 >= thres_d):
#            return 'win'
#        else:
#            return 'draw'       
    if (diff_w >= thres_w ):
        return 'lose'
    elif (diff_w <= -thres_w ):
        return 'win'
    else:
        return 'draw'
    
def compare(row):
    if(row['result'] == row['prediction']):
        return 1
    else:
        return 0
    
    
data_ = data.copy()    
data_['prediction'] = data_.apply(infer_result, axis=1)

#data_['compare'] = data_.apply(compare, axis=1)

acc = data_.loc[data_['prediction'] == data_['result'],:]
acc.shape[0] / data_.shape[0]

from collections import Counter

prediction = Counter(acc['result'])
result = Counter(data_['result'])
prediction
result



