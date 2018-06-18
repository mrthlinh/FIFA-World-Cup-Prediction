# -*- coding: utf-8 -*-

# Scrap web
from bs4 import BeautifulSoup
import requests
#import pandas as pd
import sys

def retrieve_info(r,verbose=False):
#    r = requests.get(url)
#    global df_index
#    global squad_info
    html_doc = r.text
    soup = BeautifulSoup(html_doc,"lxml")
    #print(soup.title)

    # Find the table of players
#    table = soup.find('table', class_='table table-striped players')
    # Extract url to player info
    player_link = soup.find_all('td', attrs={"data-title": "Name"})
    # Build list of full url
    inner_link = [url_home+link.a['href'] for link in player_link] # List compprehension

    # Iterate through the list of player and get info
#    squad_info = {}
#    s_link = inner_link[0:10]
#    index = 0
    for link in inner_link:
        sub_html_doc = requests.get(link).text
        sub_soup = BeautifulSoup(sub_html_doc,"lxml")


        tables = sub_soup.findAll('div', attrs={"class": "panel panel-info"})


        player_info = {}
        country = sub_soup.find('h2',attrs={"class": "subtitle"}).get_text()
        player_info['Country'] = country
        
        header = tables[0].find('h3',attrs={"class": "panel-title"}).get_text()
        _ = header.split(" ")
        player_info['Name'] = ' '.join(_[0:-2])

        print(player_info['Name'])




url_home = 'https://www.fifaindex.com'
versions = ['fifa05_1','fifa06_2','fifa07_3','fifa08_4','fifa09_5','fifa10_6',
'fifa11_7','fifa12_9','fifa13_11','fifa14_13','fifa15_14']

version_id = int(sys.argv[1]) - 5
#version_id = 1
version = versions[version_id]
print("Version of FIFA: ",version)

page_num = sys.argv[2]
#page_num = "50"
url = url_home + '/players/'+version+'/'+page_num
r = requests.get(url)
print(r.status_code)

retrieve_info(r,verbose=True)
