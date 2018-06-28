# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:46:14 2018

@author: mrthl
"""

# Scrap web
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from helper_function import convert_moneyline_decimal
import sys
import re
#============================== Read command line ===================================================
def doc():
    '''
Description: Update the match data (csv) with new matches defined in a txt file
Run: python updater.py <data_path> <url_list_path> <replace>
    <data_path> path to existing match data
    <url_list_path> path to url list (tab delimited, <url> tab <league name>)
    <replace> boolean T (default) or F whether replace existing file or not
Example: python updater.py data/database_matches.csv url_list/url_list.txt F

    '''
    print (doc.__doc__)

doc()
arg_len = len(sys.argv)
#print(arg_len)
if (arg_len > 2 & arg_len <5):
    data_path = sys.argv[1]
    url_list_path = sys.argv[2]
    replace = True
    if (arg_len > 3):
        replace_arg = sys.argv[3]
        if replace_arg == 'T':    
            replace = True
        elif replace_arg == 'F':
            replace = False
        else:
            print("Invalid command")
            quit()
else:
    print("Invalid command")
    quit()
    
#============================== Read files ===================================================
url_list = []
#with open('url_list.txt') as f:
with open(url_list_path) as f:
#    pair = []
    
    for item in f:
        if (item != ''):
            url_list.append(item.split('\t'))

#database = pd.read_csv("data/database_matches.csv")
database = pd.read_csv(data_path)

#=============================== Main functions  ==================================================
#path = "C:/Users/mrthl/geckodriver/geckodriver.exe"
driver = webdriver.Firefox()
for pair in url_list:
    url = pair[0].strip()
    league = pair[1].strip()
    
    max_page = int(pair[2].strip())
    print('========================================================')
    print("Reading URL: ",url)
    print("League: ",league)
    print("Max Page: ",max_page)
    
#    check_name = league in database['league'].unique()
#    if (~check_name):
#        quit()
     
    records = []
    labels = ['match_id','league','match_date','team_1','score_1','team_2','score_2','note','avg_odds_win_1','avg_odds_draw','avg_odds_win_2']
    
#    Loop over all pages 
    for idx in range(max_page):
        print("------------------ Page ",str(idx+1),"------------------")
        url_full = url + "#/page/" + str(idx + 1) + "/"
        print(url_full)
        driver.get(url_full)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        matches = soup.findAll('td',class_="name table-participant")         
        sections = soup.findAll('tr',class_="center nob-border")
        match_id = database.iloc[-1,0] + len(matches)
        league_full = league
        
        # Loop over all sub sections
        for i in range(len(sections)):
            section = sections[i]
            
            try:
                next_section = sections[i+1]
            except:
                print("============= Last One ==============")
                next_section = sections[0]
                
            while ((section != None) & (section != next_section)):
                attr = section.attrs['class']
                
                # Get match date
                if ('center' in attr):
                    header = section.get_text().split(' - ')
                    date = header[0].replace("1X2B's",'')
                    print('.........',date,'.........')
                    if (len(header) > 1):
                        qualification = header[1].replace("1X2B's",'')
                        if (qualification == 'Qualification'):
                            league_full = league + " " + qualification

                # Get match details
                elif ('deactivate' in attr):
                    time = section.find('td',class_='table-time').get_text()
                    full_time = date + " " +time
                    
                    both_team = section.find('td',class_='table-participant').get_text().split('-')
                    team1 = both_team[0].strip()
                    team2 = both_team[1].strip() 
                    print(team1, '-',team2)
                    match_result = section.find('td',class_='table-score').get_text()
                    match_result = match_result.replace(u'\xa0', u' ').split(' ')
                    goal_t1 = ''
                    goal_t2 = ''
                    note = ''
                    
                    if (len(match_result) == 2):
                        note = ' '.join(match_result)
                    else:
                        s = match_result[0]
                        s_format = re.findall("[0-9]:[0-9]",s)
                        if (len(s_format) == 0):
                            note = s
                        else:
    #                        match_result = match_result[0].split(':')
                            match_result = s.split(':')
                            goal_t1 = match_result[0]
                            goal_t2 = match_result[1]      
                    
                    odds = section.findAll('td',class_='odds-nowrp')        
                    odd_t1 = convert_moneyline_decimal(odds[0].get_text())
                    odd_d  = convert_moneyline_decimal(odds[1].get_text())
                    odd_t2 = convert_moneyline_decimal(odds[2].get_text())
                    
                    try:
                        datetime_object = datetime.strptime(full_time, '%d %b %Y %H:%M')
                        date_st = datetime.strftime(datetime_object,'%m/%d/%Y %H:%M')
                    except:
#                        print("Error :",full_time)
                        st = date.split(', ')
                        if (st[0] == 'Yesterday') | (st[0] == 'Today'):
                            error_date = date.split(', ')[1] + str(datetime.today().year)
                            full_time = error_date + " " +time
                            datetime_object = datetime.strptime(full_time, '%d %b %Y %H:%M')
                            date_st = datetime.strftime(datetime_object,'%m/%d/%Y %H:%M')
                        else:
                            break

                    
                    result_text = "{} {}-{} {}-{} {} {}-{}-{}".format(date_st,team1,team2,goal_t1,goal_t2,note,odd_t1,odd_d,odd_t2)
                    
                    match_id = match_id - 1
                    
                    records.append([match_id,league_full,date_st,team1,goal_t1,team2,goal_t2,note,odd_t1,odd_d,odd_t2])
                    
    #                print(result_text) 
    #                
                section = section.findNextSibling()
                            
                           
        driver.execute_script("window.open('');")
        driver.close()
#        driver.switch_to.window(driver.window_handles[idx+1])
        driver.switch_to.window(driver.window_handles[0])
        
    df = pd.DataFrame.from_records(records,columns=labels)    
    database = database.append(df)
        
driver.quit()

#=============================== Save file ==================================================
database['match_date'] = pd.to_datetime(database.match_date)
database = database.sort_values(by='match_date')
database['match_id'] = list(range(1,database.shape[0]+1))
if (replace):
    database.to_csv(data_path, index = False)
else:
    filename = data_path.split('.')[0]
    database.to_csv(filename+"_new.csv", index = False)
