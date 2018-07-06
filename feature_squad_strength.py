# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:42:11 2018

@author: mrthl
"""
#======================================== Load Library ================================================
# Load library
import pandas as pd
#import numpy as np
import datetime as dt
import sys
#============================== Read command line ===================================================
def doc():
    '''
Description: Get the head-to-head and form of recent N matches of both teams

Run: python feature_h2h_form.py <path all matches> <path new matches> <replace>
    <data_path> path to existing match data
    <url_list_path> path to url list (tab delimited, <url> tab <league name>)
    <replace> boolean T (default) or F whether replace existing file or not
    
Example: python updater.py data/database_matches.csv url_list/url_list.txt F

    '''
    print (doc.__doc__)

doc()

#arg_len = len(sys.argv)
##print(arg_len)
#if (arg_len > 2 & arg_len <5):
#    new_matches_str = sys.argv[1]
#    squad_strength_str = sys.argv[2]
#else:
#    print("Invalid command")
#    quit()


#all_matches_str = "data/raw/international-football-results-from-1872-to-2018.csv"
new_matches_str = "data/oddportal/2018_WC_odds_quarter_h2h_form_.csv"
squad_strength_str = "data/squad_strength/2018_FIFA_World_Cup_squads_strength__.csv"
#num_recent_matches = 10

#======================================== Load Data ===================================================

# History file (main database)
#all_match = pd.read_csv("data/raw/international-football-results-from-1872-to-2018.csv")

# New matches which we have odd already
new_matches = pd.read_csv(new_matches_str)

ss_wc_18 = pd.read_csv(squad_strength_str, encoding='utf-8')
ss_wc_18 = ss_wc_18.set_index(ss_wc_18['Nation'])


#================================================================================================================
new_features_game= ["game_diff_rank","game_diff_ovr",
               "game_diff_attk","game_diff_mid",
               "game_diff_def","game_diff_prestige",
               "game_diff_age11","game_diff_ageAll",
               "game_diff_bup_speed","game_diff_bup_pass",
               "game_diff_cc_pass","game_diff_cc_cross",
               "game_diff_cc_shoot","game_diff_def_press",
               "game_diff_def_aggr","game_diff_def_teamwidth"]

squad_features = ['Rank','Overall',
                 'Attack','Midfield',
                  'Defence','InternationalPrestige',
                  'Starting11AverageAge','WholeTeamAverageAge',
                 'BuildUpPlay_Speed', 'BuildUpPlay_Passing',
                 'ChanceCreation_Passing', 'ChanceCreation_Crossing',
                 'ChanceCreation_Shooting', 'Defence_Pressure', 
                 'Defence_Aggression', 'Defence_TeamWidth']


def diff_feature(row,df):
    team1= row.team_1
    team2= row.team_2
    rank1 = df.loc[team1]
    rank2 = df.loc[team2]
    return (rank1-rank2)

        
def addFeature_game(new_features_game,squad_features,df_event,df_squad):    
    for i in range(len(new_features_game)):
        new_feature = new_features_game[i]
        squad_feature = squad_features[i]
        df_event[new_feature] = df_event.apply(diff_feature,df = df_squad[squad_feature],axis=1)
    return df_event
       
data_WC_2018 = addFeature_game(new_features_game,squad_features,new_matches,ss_wc_18)  

data_WC_2018.to_csv('data/data_full_feature/data_WC_2018_quarter.csv',index=False)
#====================================== Save file ==============================================
filename = new_matches_str.split('.')[0] +"_squad_strength.csv"
new_matches.to_csv(filename, index = False)
print("Create new file: ",filename)

