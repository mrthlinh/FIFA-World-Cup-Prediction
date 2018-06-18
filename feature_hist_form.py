# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:22:51 2018

@author: mrthl
"""
# Load library
#import csv
import pandas as pd
import numpy as np
#import time
#from datetime import datetime, timedelta

all_match = pd.read_csv("data/raw/international-football-results-from-1872-to-2018.csv")
wc_match = pd.read_csv("data/2018_WC_odds.csv")


def w_d_l(row):
    if row > 0:
        return 'win'
    elif row == 0:
        return 'draw'
    else:
        return 'lose'
    
def correct_Korea(st):
    if st == "South Korea":
        return "Korea Republic"
    else:
        return st
    
def hist_feature(row):        
    team1= correct_Korea(row.team_1)
    team2= correct_Korea(row.team_2)
    
    df_1 = all_match.loc[(all_match['home_team'] == team1) & (all_match['away_team'] == team2),'home_score':'away_score']     
    df_2 = all_match.loc[(all_match['home_team'] == team2) & (all_match['away_team'] == team1),'home_score':'away_score']
    df_2.columns = ['away_score','home_score']
    df_3 = pd.concat([df_1,df_2])
    diff = df_3['home_score'] - df_3['away_score']
    result = diff.apply(w_d_l)
    result = result.value_counts()
    
    num_win = result.get('win',0)
    num_draw = result.get('draw',0)
    num_lose = result.get('lose',0)
    
    diff_win = num_win - num_lose
    
    return [diff_win,num_draw]

def form_feature(team,num_recent_matches):
    df = all_match.loc[(all_match['home_team'] == team) | (all_match['away_team'] == team),'date':'away_score']
    df = df.tail(num_recent_matches)
    
    df_home = df.loc[df['home_team'] == team,'home_score':'away_score']
    df_away = df.loc[df['away_team'] == team,'home_score':'away_score']
    df_away.columns = ['away_score','home_score']
    df_sum = pd.concat([df_home,df_away])    
    goal = df_sum.apply(np.sum,axis=0) 
    
    diff = df_sum['home_score'] - df_sum['away_score']
    result = diff.apply(w_d_l)
    result = result.value_counts()
    
    f_goalF = goal['home_score']
    f_goalA = goal['away_score']
    f_win   = result.get('win',0)
    f_draw   = result.get('draw',0)
    
    return {"f_goalF":f_goalF,"f_goalA":f_goalA,
            "f_win":f_win, "f_draw":f_draw }
  
def form_10(row):
    
    team1= correct_Korea(row.team_1)
    team2= correct_Korea(row.team_2)
    num_recent_matches = 10
    
    team1_info = form_feature(team1,num_recent_matches)
    team2_info = form_feature(team2,num_recent_matches)
    
    return [team1_info['f_goalF'],team2_info['f_goalF'],team1_info['f_goalA'],team2_info['f_goalA'],
            team1_info['f_win'],team2_info['f_win'],team1_info['f_draw'],team2_info['f_draw']]
      
      
wc_match['h_win_diff'], wc_match['h_draw']= zip(*wc_match.apply(hist_feature,axis=1))
wc_match['f_goalF_1'], wc_match['f_goalF_2'], wc_match['f_goalA_1'], wc_match['f_goalA_2'],wc_match['f_win_1'], wc_match['f_win_2'], wc_match['f_draw_1'], wc_match['f_draw_2'] = zip(*wc_match.apply(form_10,axis=1))

wc_match.to_csv("data/data_WC_2018.csv",index=False)

# Check if there is some missing
print(np.sum(wc_match.isnull()))




