# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:51:19 2018

@author: mrthl
"""

# Scrap web
from bs4 import BeautifulSoup
#import requests
import pandas as pd
#import sys
#from unidecode import unidecode

f = open('C:/Users/mrthl/Desktop/betodd/odd.html',encoding="utf-8" )
data = f.read()
soup = BeautifulSoup(data, 'html.parser')
matches = soup.findAll('td',class_="name table-participant")
odds = soup.findAll('td', class_="odds-nowrp")

def convert_moneyline_decimal(st):
    num = float(st)
    result = 0
    if num > 0:
        result = num / 100 + 1
    elif num < 0:
        result = -100 / num + 1
    return round(result,2)
    
records = []
labels = ['team_1','team_2','avg_odds_win_1','avg_odds_draw','avg_odds_win_1']

for i in range(len(matches)):
#    record = []
    
    both_team = matches[i].get_text().split('-')
    team1 = both_team[0].strip()
    team2 = both_team[1].strip()
    
    odd_t1 = convert_moneyline_decimal(odds[3*i].get_text())
    odd_d  = convert_moneyline_decimal(odds[3*i + 1].get_text())
    odd_t2 = convert_moneyline_decimal(odds[3*i +2].get_text())
    
    records.append([team1,team2,odd_t1,odd_d,odd_t2])
    
    
    odd = "{} - {} - {}".format(odd_t1,odd_d,odd_t2)
    print(matches[i].get_text(),': ',odd)

df = pd.DataFrame.from_records(records,columns=labels)

df.to_csv("data/2018_WC_odds.csv",index = False)

