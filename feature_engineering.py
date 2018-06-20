# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:30:40 2018

@author: mrthl
"""
# Load library
import pandas as pd
#import numpy as np
import datetime as dt

#================================================================================================================
# Read data full 2005 - 2015
data_2005 = pd.read_csv("data/data_odd_2005_regression.csv", encoding='utf-8')
data_2016 = pd.read_csv("data/data_EURO_2016.csv", encoding='utf-8')
#data_2018 = pd.read_csv("data/data_WC_2018.csv", encoding='utf-8')
data_2018 = pd.read_csv("data/data_WC_2018_round2.csv", encoding='utf-8')
# Convert date to Datetime object
data_2005['date'] = pd.to_datetime(data_2005['date'])
data_2016['date'] = pd.to_datetime(data_2016['date'])

original_col = list(data_2005.columns)
result_col = original_col[-2:]
new_col = original_col[0:8]

# Read data squad strength
ss_wc_10 = pd.read_csv("web_crawler/squad/squad_strength/2010_FIFA_World_Cup_squads_strength__.csv", encoding='utf-8')
ss_wc_14 = pd.read_csv("web_crawler/squad/squad_strength/2014_FIFA_World_Cup_squads_strength__.csv", encoding='utf-8')
ss_eu_12 = pd.read_csv("web_crawler/squad/squad_strength/2012_UEFA_Euro_squads_strength__.csv", encoding='utf-8')
ss_eu_16 = pd.read_csv("web_crawler/squad/squad_strength/2016_UEFA_Euro_squads_strength__.csv", encoding='utf-8')
ss_wc_18 = pd.read_csv("web_crawler/squad/squad_strength/2018_FIFA_World_Cup_squads_strength__.csv", encoding='utf-8')

ss_wc_10 = ss_wc_10.set_index(ss_wc_10['Nation'])
ss_wc_14 = ss_wc_14.set_index(ss_wc_14['Nation'])
ss_eu_12 = ss_eu_12.set_index(ss_eu_12['Nation'])
ss_eu_16 = ss_eu_16.set_index(ss_eu_16['Nation'])
ss_wc_18 = ss_wc_18.set_index(ss_wc_18['Nation'])

#================================================================================================================
# Take difference of existing features

new_features= ["form_diff_goalF","form_diff_goalA",
               "form_diff_win","form_diff_draw",
               "odd_diff_win","odd_draw"]

old_features = ['f_goalF_','f_goalA_',
                'f_win_','f_draw_',
                'avg_odds_win_','avg_odds_draw']

new_col = new_col+new_features

def addFeature_diff(new_features,old_features,my_df,drop=True):  
    df = my_df.copy()
    for i in range(len(new_features)-1):
        new_feature = new_features[i]
        old_feature_1 = old_features[i] + "1"
        old_feature_2 = old_features[i] + "2"
        
        df[new_feature] = df.apply(lambda row: (row[old_feature_1] - row[old_feature_2]),axis=1)
        if (drop):
            df = df.drop([old_feature_1,old_feature_2],axis=1)
#    'avg_odds_draw' -> no need to take difference
    df[new_features[-1]] = my_df[old_features[-1]]
    if (drop):
        df = df.drop([old_features[-1]],axis=1)
    return df

data_2005_new = addFeature_diff(new_features,old_features,data_2005,drop=True)
data_2016_new = addFeature_diff(new_features,old_features,data_2016,drop=True)
data_2018_new = addFeature_diff(new_features,old_features,data_2018,drop=True)


#================================================================================================================

data_WC_2010_ = data_2005_new[(data_2005_new['tournament'] == 'FIFA World Cup') & (data_2005_new['date'].dt.year == 2010 )]
list_nation_10 = ss_wc_10['Nation'].unique()
data_WC_2010 = data_WC_2010_[(data_WC_2010_['team_1'].isin(list_nation_10)) & (data_WC_2010_['team_2'].isin(list_nation_10))]

_ = data_WC_2010_.index.difference(data_WC_2010.index)
diff_10 = data_WC_2010_.loc[_]

data_WC_2014_ = data_2005_new[(data_2005['tournament'] == 'FIFA World Cup') & (data_2005_new['date'].dt.year == 2014 )]
list_nation_14 = ss_wc_14['Nation'].unique()
data_WC_2014 = data_WC_2014_[(data_WC_2014_['team_1'].isin(list_nation_14)) & (data_WC_2014_['team_2'].isin(list_nation_14))]

_ = data_WC_2014_.index.difference(data_WC_2014.index)
diff_14 = data_WC_2014_.loc[_]

data_EUR_2012_ = data_2005_new[(data_2005['tournament'] == 'UEFA Euro') & (data_2005_new['date'].dt.year == 2012 )]
list_nation_12 = ss_eu_12['Nation'].unique()
data_EUR_2012 = data_EUR_2012_[(data_EUR_2012_['team_1'].isin(list_nation_12)) & (data_EUR_2012_['team_2'].isin(list_nation_12))]

_ = data_EUR_2012_.index.difference(data_EUR_2012.index)
diff_12 = data_EUR_2012_.loc[_]

data_EUR_2016_ = data_2016_new.copy()
list_nation_16 = ss_eu_16['Nation'].unique()
data_EUR_2016 = data_EUR_2016_[(data_EUR_2016_['team_1'].isin(list_nation_16)) & (data_EUR_2016_['team_2'].isin(list_nation_16))]

_ = data_EUR_2016_.index.difference(data_EUR_2016.index)
diff_16 = data_EUR_2016_.loc[_]

data_WC_2018 = data_2018_new.copy()


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

new_col = new_col + new_features_game + result_col

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

_=addFeature_game(new_features_game,squad_features,data_WC_2010,ss_wc_10)
_=addFeature_game(new_features_game,squad_features,data_WC_2014,ss_wc_14)
_=addFeature_game(new_features_game,squad_features,data_EUR_2012,ss_eu_12)
_=addFeature_game(new_features_game,squad_features,data_EUR_2016,ss_eu_16)        
_=addFeature_game(new_features_game,squad_features,data_WC_2018,ss_wc_18)  

 

data_WC_2010 = data_WC_2010[new_col]
data_WC_2014 = data_WC_2014[new_col]
data_EUR_2012 = data_EUR_2012[new_col]
data_EUR_2016 = data_EUR_2016[new_col]

data_WC_2010.to_csv('data/data_full_feature/data_WC_2010.csv',index=False)
data_WC_2014.to_csv('data/data_full_feature/data_WC_2014.csv',index=False)


data_EUR_2012.to_csv('data/data_full_feature/data_EUR_2012.csv',index=False)
data_EUR_2016.to_csv('data/data_full_feature/data_EUR_2016.csv',index=False)

# Round 2
data_WC_2018.to_csv('data/data_full_feature/data_WC_2018_round2.csv',index=False)
#
#squad_features = ['f_goalF_',
#                  'f_goalA_',
#                  'f_win_',
#                  'f_draw_',
#                  'avg_odds_win_',
#                  'avg_odds_draw']




