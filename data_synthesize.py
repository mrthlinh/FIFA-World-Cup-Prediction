# -*- coding: utf-8 -*-
"""
Created on Sat May 12 11:04:31 2018

@author: mrthl
"""

import pandas as pd
import numpy as np

# Load Data
data = pd.read_csv("data/data_odd_2005_regression.csv", encoding='utf-8')
#add_data = data.iloc[0,:]
#data_new=data.append(add_data)
length = data.shape[0]
for i in range(length):
#for i in range(20):
    print(i)
    last_id = data.iloc[-1,:].id
    add_data = data.iloc[i,:]
    myswap(add_data,['team_1','f_goalF_1','f_goalA_1','f_win_1','f_draw_1','avg_odds_win_1'],['team_2','f_goalF_2','f_goalA_2','f_win_2','f_draw_2','avg_odds_win_2'])
    reverse(add_data,['h_win_diff','goal_diff'])
    reverse_result(add_data)
    add_data.id = last_id + 1
    data = data.append(add_data)
    
def myswap(df,col1,col2):
    for i in range(len(col1)):
        col_name1= col1[i]
        col_name2= col2[i]
        
        temp = df[col_name1]
        df[col_name1] = df[col_name2]
#        df.loc[:,col_name1] = df.loc[:,col_name2]
        df[col_name2] = temp
#        df.loc[:,col_name2] = temp
        
def reverse(df,col):
    for i in range(len(col)):
        col_name = col[i]
        if (df[col_name] != 0):
            df[col_name] = -df[col_name]
        
def reverse_result(df):
    value = df['result']
    if (value == 'win'):
        df['result'] = 'lose'
    if (value == 'lose'):
        df['result'] = 'win'
        
data.to_csv('data/data_odd_2005_regression_syn.csv',index=None)
#data = data_.iloc[:,2:].copy()


