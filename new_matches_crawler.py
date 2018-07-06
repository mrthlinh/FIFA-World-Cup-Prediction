# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:51:19 2018

@author: mrthl
"""

# Scrap web
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from helper_function import convert_moneyline_decimal
import sys

#============================== Read command line ===================================================
def doc():
    '''
Description: Get team name and odd ratio for new upcoming matches prediction
Run: python new_matches_crawler.py <input path> <output path>
    <input path> path to txt file containing URL
    <output path> path to output file

Example: python new_matches_crawler.py url_list/new_match_prediction.txt data/oddportal/2018_WC_odds_round16.csv

    '''
    print (doc.__doc__)

doc()
#
arg_len = len(sys.argv)

if (arg_len == 3):
    input_path = sys.argv[1]
    output_path = sys.argv[2]

else:
    print("Invalid command")
    quit()
    
#============================== Read files ===================================================
input_path = "url_list/new_match_prediction.txt"

with open(input_path) as f:
    for item in f:
        print(item)
        url = item
   
#============================== Scrap Website ===================================================
#url = "http://www.oddsportal.com/soccer/world/world-cup-2018/"

driver = webdriver.Firefox()
driver.get(url)

soup = BeautifulSoup(driver.page_source, 'html.parser')
matches = soup.findAll('td',class_="name table-participant")
odds = soup.findAll('td', class_="odds-nowrp")
result = soup.findAll('td',class_="center bold table-odds table-score")
    
records = []
labels = ['team_1','team_2','avg_odds_win_1','avg_odds_draw','avg_odds_win_2']

max_iter = len(matches)
#max_iter = 32
for i in range(max_iter):
    
    both_team = matches[i].get_text().split('-')
    team1 = both_team[0].strip()
    team2 = both_team[1].strip()
        
    odd_t1 = convert_moneyline_decimal(odds[3*i].get_text())
    odd_d  = convert_moneyline_decimal(odds[3*i + 1].get_text())
    odd_t2 = convert_moneyline_decimal(odds[3*i +2].get_text())
    
    records.append([team1,team2,odd_t1,odd_d,odd_t2])
    
    
    result_text = "{} - {} - {}".format(odd_t1,odd_d,odd_t2)
    print(matches[i].get_text(),'\t',result_text)

driver.quit()
df = pd.DataFrame.from_records(records,columns=labels)

df.to_csv(output_path,index = False)
print("Create new file: ",output_path)