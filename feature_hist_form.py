# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:22:51 2018

@author: mrthl
"""
#======================================== Load Library ================================================
import pandas as pd
import numpy as np
import sys
from helper_function import hist_feature,form_feature,form_feature_new,hist_feature_new

#============================== Read command line ===================================================
def doc():
    '''
Description: Get the head-to-head and form of recent N matches of both teams

Run: python feature_hist_form.py <path all matches> <path new matches> <number of recent matches> <take difference>
    <path all matches> path to existing match data
    <path new matches> 
    <number of recent matches>
    <diff>
    
Example: python feature_hist_form.py data/oddportal/2018_WC_odds_round16.csv 10 T

Future Implementation: Add Date to <new matches> and Merge Database
    '''
    print (doc.__doc__)

doc()

arg_len = len(sys.argv)
if (arg_len == 4):
    new_matches_str = sys.argv[1]
    num_recent_matches = int(sys.argv[2])
    take_diff = bool(sys.argv[3])
else:
    print("Invalid command")
    quit()


#all_matches_str = "data/raw/international-football-results-from-1872-to-2018.csv"
all_matches_str = "data/football-matches-1872-to-2018.csv"
new_matches_str = "data/oddportal/2018_WC_odds_quarter.csv"
num_recent_matches = 10
take_diff = True
#======================================== Load Data ===================================================

# History file (main database)
all_match = pd.read_csv(all_matches_str)

# New matches which we have odd already
new_matches = pd.read_csv(new_matches_str)
df = new_matches.iloc[:,:2]
#======================================== New Features ================================================

#               "odd_diff_win","odd_draw"]
#unpack_form = list(zip(*new_matches.apply(form_feature,df_hist = all_match,num_recent_matches = num_recent_matches,bool_diff = take_diff,axis=1)))
#unpack_hist = list(zip(*new_matches.apply(hist_feature,df_hist = all_match,bool_diff = take_diff, axis=1)))

unpack_form = list(zip(*new_matches.apply(form_feature_new,df_hist = all_match,num_recent_matches = num_recent_matches,bool_diff = take_diff,axis=1)))
unpack_hist = list(zip(*new_matches.apply(hist_feature_new,df_hist = all_match,bool_diff = take_diff, axis=1)))

if (take_diff):
    df['h2h_win_diff'], df['h2h_draw'] = unpack_hist  
    df['form_diff_goalF'], df['form_diff_goalA'], df['form_diff_win'], df['form_diff_draw'] = unpack_form
    df["odd_win_diff"] = new_matches['avg_odds_win_1'] - new_matches['avg_odds_win_2'] 
    df["odd_draw"] = new_matches['avg_odds_draw']
    annotation = "_h2h_form.csv"
else:
    df['h2h_num_win'], df['h2h_num_lose'], df['h2h_draw'] = unpack_hist 
    df['form_goalF_1'], df['form_goalF_2'], df['form_goalA_1'], df['form_goalA_2'],df['form_win_1'], df['form_win_2'], df['form_draw_1'], df['form_draw_2'] = unpack_form       
    annotation = "_h2h_form_raw.csv"



#======================================== Visualization ================================================



#======================================== Check ================================================
# Check if there is some missing
print("========= Check Null =========")
check = np.sum(df.isnull())
if (np.sum(check) == 0):
    print("No Missing Value")
    
else:
    print(check)

#====================================== Save file ==============================================
filename = new_matches_str.split('.')[0] + annotation
df.to_csv(filename, index = False)
print("Create new file: ",filename)

