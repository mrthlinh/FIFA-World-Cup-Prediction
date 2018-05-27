# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:19:39 2018

@author: mrthl
"""

# Convert classfication -> regression

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

data = pd.read_csv("data/data_odd_2005.csv", encoding='utf-8')
data_full = pd.read_csv("data/data_odd_match_filter.csv", encoding='utf-8')
data['date'] = pd.to_datetime(data['date'])
data_full['match_date'] = pd.to_datetime(data_full['match_date'])
# Fill the goal differences

wrong = list()
for i in range(data.shape[0]):
#for i in range(1793,1794):
    print(i)
    curr_match = data.iloc[i,:] #data with odd

    lookup_match = data_full[(curr_match.date == data_full.match_date) & (curr_match.home_team == data_full.home_team)]
    if (lookup_match.shape[0] ==0):
        lookup_match = data_full[(curr_match.date == data_full.match_date) & (curr_match.home_team == data_full.away_team)]
#        if (lookup_match.shape[0] == 1 ): #Correct home team name
#            data.loc[i,'home_team'] = lookup_match.home_team       
    if (lookup_match.shape[0] ==0):
        lookup_match = data_full[(curr_match.date == data_full.match_date + timedelta(days=-1)) & (curr_match.home_team == data_full.home_team)]
        if (lookup_match.shape[0] == 0):
            lookup_match = data_full[(curr_match.date == data_full.match_date + timedelta(days=-1)) & (curr_match.home_team == data_full.away_team)]
#            if (lookup_match.shape[0] == 0): #Correct home team name
#                data.loc[i,'home_team'] = lookup_match.home_team
    if (lookup_match.shape[0] ==0):
        lookup_match = data_full[(curr_match.date == data_full.match_date + timedelta(days=+1)) & (curr_match.home_team == data_full.home_team)]
        if (lookup_match.shape[0] == 0):
            lookup_match = data_full[(curr_match.date == data_full.match_date + timedelta(days=+1)) & (curr_match.home_team == data_full.away_team)]
#            if (lookup_match.shape[0] == 0): #Correct home team name
#                data.loc[i,'home_team'] = lookup_match.home_team       
    if (lookup_match.shape[0] > 1 ):
        lookup_match = lookup_match.iloc[0,:]
        data.loc[i,'home_team'] = lookup_match.home_team
    lookup_match = lookup_match.squeeze()
    count = 0
#    Add goal difference to "data with odd"
    print("{} -- {} :".format(curr_match.team_1,curr_match.team_2))
    try:
        
        if curr_match.team_1 == lookup_match.home_team:
            goal_diff = lookup_match.home_score - lookup_match.away_score
            print("{} -- {} :".format(lookup_match.home_score, lookup_match.away_score))
            print("goal difference: ",goal_diff)
            data.loc[i,'goal_diff'] = goal_diff
        elif curr_match.team_2 == lookup_match.home_team:
            goal_diff = -lookup_match.home_score + lookup_match.away_score
            print("{} -- {} :".format(lookup_match.away_score, lookup_match.home_score))
            print("goal difference: ",goal_diff)
            data.loc[i,'goal_diff'] = goal_diff
        else:
            print("there is something wrong here")
            print(curr_match)
            count = count + 1
    except:
        count = count + 1
        wrong.append(i)
print("something wrong = ",wrong)
data.to_csv('data/data_odd_2005_regression.csv',index=None)
    
