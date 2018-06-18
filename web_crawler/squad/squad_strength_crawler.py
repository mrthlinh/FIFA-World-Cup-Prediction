# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:47:43 2018

@author: mrthl
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
import sys
from bs4 import UnicodeDammit
from unidecode import unidecode

def retrieve_info(url,test=False):
    r = requests.get(url)
#    status = r.status_code
    html_doc = r.text
    soup = BeautifulSoup(html_doc,"lxml")
    
    labels = ['Nation']
    team_info = []
    
    nation_name = soup.find('h1',class_="media-heading").get_text()
    nation_name = unidecode(nation_name)
    
    team_info.append(nation_name)
    
    tables = soup.findAll('div',class_="panel panel-info")
    # We dont need the second and the last table so delete them
    _ = tables.pop(1)
    _ = tables.pop(-1)
    
    #for i in range(len(tables)):
    #    print(tables[i].find('h3').get_text())

    table_idx = 0
    for table in tables:
        suffix = table.find('h3').get_text().strip().replace(' ','')
        suffix += '_'
        
        print(suffix)
        
        rows = table.findAll(['li','p'])
        
        if table_idx == 0:
            attr_name = rows[0].contents[0].strip().replace(' ','')
            attr_value= rows[0].contents[1].get_text()
            attr_value = unidecode(attr_value)
            
            labels.append(suffix+attr_name)
            team_info.append(attr_value)
            
            print(attr_name, "-",attr_value)
            for row in rows[1:-1]:
                attr_name = row.contents[2].strip()
                attr_value= row.contents[1].get_text()
                attr_value = unidecode(attr_value)
                
                labels.append(suffix+attr_name)
                team_info.append(attr_value)
                
                print(attr_name, "-",attr_value)
        else:
            for row in rows:
                attr_name = row.contents[0].strip().replace(' ','')
                attr_value= row.contents[1].get_text()
                attr_value = unidecode(attr_value)
                
                if (test == False):
                    labels.append(suffix+attr_name)
                    team_info.append(attr_value)
                elif (attr_name != 'Dribbling'):
                    labels.append(suffix+attr_name)
                    team_info.append(attr_value)
                
#                labels.append(suffix+attr_name)
#                team_info.append(attr_value)
                    
                print(attr_name, "-",attr_value)
        
        table_idx += 1
        
    return [labels,team_info]
    
#url_home = 'https://www.fifaindex.com'
#versions = ['fifa05_1','fifa06_2','fifa07_3','fifa08_4','fifa09_5','fifa10_6',
#'fifa11_7','fifa12_9','fifa13_11','fifa14_13','fifa15_14','fifa16_73','fifa17_173','']


#url = "https://www.fifaindex.com/team/1337/germany/fifa11_7/"

#https://www.fifaindex.com/team/1330/czech-republic    
#https://www.fifaindex.com/team/1331/denmark/


#===================================================================
# WC 10
    #Nigeria, Algeria,Ghana,Serbia,Japan,Slovakia,North Korea,Honduras
#version = 'fifa11_7'
#list_nation = ["https://www.fifaindex.com/team/1335/france/",
#        "https://www.fifaindex.com/team/1386/mexico/",
#        "https://www.fifaindex.com/team/111099/south-africa/",
#        "https://www.fifaindex.com/team/1377/uruguay/",
#        "https://www.fifaindex.com/team/1369/argentina/",
#        "https://www.fifaindex.com/team/1338/greece/",
#        "https://www.fifaindex.com/team/974/korea-republic/",
#        "https://www.fifaindex.com/team/1318/england/",
#        "https://www.fifaindex.com/team/1361/slovenia/",
#        "https://www.fifaindex.com/team/1387/united-states/",
#        "https://www.fifaindex.com/team/1415/australia/",
#        "https://www.fifaindex.com/team/1337/germany/",
#        "https://www.fifaindex.com/team/1395/cameroon/",
#        "https://www.fifaindex.com/team/1331/denmark/",
#        "https://www.fifaindex.com/team/105035/netherlands/",
#        "https://www.fifaindex.com/team/1343/italy/",
#        "https://www.fifaindex.com/team/111473/new-zealand/",
#        "https://www.fifaindex.com/team/1375/paraguay/",
#        "https://www.fifaindex.com/team/1370/brazil/",
#        "https://www.fifaindex.com/team/111112/c%C3%B4te-divoire/",
#        "https://www.fifaindex.com/team/1354/portugal/",
#        "https://www.fifaindex.com/team/111459/chile/",
#        "https://www.fifaindex.com/team/1362/spain/",
#        "https://www.fifaindex.com/team/1364/switzerland/"]
#========================================================================

#Croatia, Japan, Costa Rica, Honduras.Bosnia and Herzegovina, Iran,Nigeria,
#Ghana
#list_nation = ["https://www.fifaindex.com/team/1370/brazil/",
#               "https://www.fifaindex.com/team/1395/cameroon/",
#               "https://www.fifaindex.com/team/1386/mexico/",
#               "https://www.fifaindex.com/team/1415/australia/",
#               "https://www.fifaindex.com/team/111459/chile/",
#               "https://www.fifaindex.com/team/105035/netherlands/",
#               "https://www.fifaindex.com/team/1362/spain/",
#               "https://www.fifaindex.com/team/111109/colombia/",
#               "https://www.fifaindex.com/team/1338/greece/",
#               "https://www.fifaindex.com/team/111112/c%C3%B4te-divoire/",
#               "https://www.fifaindex.com/team/1318/england/",
#               "https://www.fifaindex.com/team/1343/italy/",
#               "https://www.fifaindex.com/team/1377/uruguay/",
#               "https://www.fifaindex.com/team/111465/ecuador/",
#               "https://www.fifaindex.com/team/1335/france/",
#               "https://www.fifaindex.com/team/1364/switzerland/",
#               "https://www.fifaindex.com/team/1369/argentina/",
#               "https://www.fifaindex.com/team/1337/germany/",
#               "https://www.fifaindex.com/team/1354/portugal/",
#               "https://www.fifaindex.com/team/1387/united-states/",
#               "https://www.fifaindex.com/team/1325/belgium/",
#               "https://www.fifaindex.com/team/1357/russia/",
#               "https://www.fifaindex.com/team/974/korea-republic/"]
    
#version = 'fifa14_13'
#=======================================================================
nation_code = {'brazil':1370, 'cameroon':1395, 'mexico':1386, 'australia':1415,
               'chile':111459, 'netherlands':105035, 'spain':1362, 'colombia':111109,
               'greece':1338, 'c%C3%B4te-divoire':111112, 'england':1318, 'italy': 1343,
               'uruguay': 1377, 'ecuador': 111465, 'france':1335, 'switzerland':1364,
               'argentina':1369, 'germany':1337, 'portugal':1354, 'united-states':1387,
               'belgium':1325, 'russia': 1357, 'korea-republic':974,'poland':1353,
               'czech-republic':1330, 'denmark':1331, 'ireland':1355, 'sweden':1363,'wales':1367,
               'northern-ireland':110081,'turkey':1365,'iceland':1341,'austria':1322, 'hungary':1886,
               'romania':1356,'egypt':111130,'south-korea':974,'saudi-arabia':111114,'peru':111108,
               'croatia':1328,'nigeria':1393,'costa-rica':1383,'serbia':110082,'tunisia':1391}

#version = 'fifa14_13'
#version = 'fifa16_59'
version = 'fifa18wc_248'
squad_file = pd.read_csv("2018_FIFA_World_Cup_squads.csv")
nation_list = squad_file['Country'].unique()

#nation_list = ['France','Romania','Albania','Switzerland','	England','Russia','Wales',
#               'Slovakia','Germany','Ukraine','Poland','Northern Ireland','Spain','Czech Republic',
#               'Turkey','Croatia','Belgium','Italy','Republic of Ireland','Sweden','Portugal','Iceland',
#               'Austria','Hungary']


nation_list = [st.lower().replace(' ','-') for st in nation_list]

# Check with countries not in FIFA Games
for n in nation_list:
    num = nation_code.get(n)
    if num == None:
        print(n)


urls = []
for nation in nation_list:
    if nation in nation_code:
        code = nation_code[nation]
        print(code,': ',nation)
        url = "https://www.fifaindex.com/team/"+str(code)+'/'+nation+'/'+version
        urls.append(url)
    if nation == 'republic-of-ireland':
        code = nation_code['ireland']
        print(code,': ireland')
        url = "https://www.fifaindex.com/team/"+str(code)+'/ireland/'+version
        urls.append(url)  
    
#urls = [n+version for n in list_nation]

records = []
labels = []
i= 0
for url in urls:
    label,record = retrieve_info(url,test=True)
    records.append(record)
    if i == 0:
        labels = label
    
#    if labels != label:
        
    print(label == labels)
    i += 1


df = pd.DataFrame.from_records(records,columns=labels)

df.to_csv("squad_strength/2018_FIFA_World_Cup_squads_strength.csv",index = False)

#======================================================================

#Get ID of each nations

#i = 1330
#for idx in range(100):
#    num = str(idx+i)
#    url = "https://www.fifaindex.com/team/"+ num +"/a"
#    r = requests.get(url)
#    status = r.status_code
#    if (status != 404):
#        html_doc = r.text
#        soup = BeautifulSoup(html_doc,"lxml")
#        
#        #labels = ['Nation']
#        #team_info = []
#        
#        nation_name = soup.find('h1',class_="media-heading").get_text()
#        nation_name = unidecode(nation_name)
#        print(num, ": ", nation_name)



