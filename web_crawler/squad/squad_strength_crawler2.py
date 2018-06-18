# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 21:57:40 2018

@author: mrthl
"""

from bs4 import BeautifulSoup
import bs4
import csv
import requests
import pandas as pd
import sys
from bs4 import UnicodeDammit
from unidecode import unidecode
 
# Get data from sofia

def retrieve_info(url,test=False):    
    r = requests.get(url)
#    status = r.status_code
    html_doc = r.text
    soup = BeautifulSoup(html_doc,"lxml")
    
    record = []
    label  = []
    try:
        team_name = soup.find("div",class_="info")
        team_name = team_name.contents[1].get_text().split(' (')
        team_name = team_name[0].strip()
    except:
        return [None,None,None]
    
    record.append(team_name)
    label.append('Nation')
    
    over_strength = soup.select("div.stats td")
    for i in range(len(over_strength)):
        feature_name = over_strength[i].contents[0].strip()
        feature_value = over_strength[i].contents[1].get_text().strip()
        label.append(feature_name)
        record.append(feature_value)
#        print(feature_name," - ",feature_value)
        

    def extractElement(mylist):
        new_list = []
        for i in range(len(mylist)):
            element = mylist[i]
            if type(element) == bs4.element.NavigableString:
                if element.strip() != '':
                    st = element.strip().replace(' ','')
                    new_list.append(st)
            else:
                st = element.get_text().strip().replace(' ','')
                new_list.append(st)
        return new_list   
            
    basic_info = soup.select("ul.pl li")
    for i in range(len(basic_info)):
        row = extractElement(basic_info[i].contents)
        feature_name = row[0]
        feature_value = row[1]
        label.append(feature_name)
        record.append(feature_value)
#        print(feature_name,"-",feature_value)       
    
    
    suffix = soup.findAll("dt")[0:3]
    suffix = [suffix[i].get_text().replace(' ','') for i in range(3) ]
    
    attr = soup.findAll("dd")[:12]
    for i in range(len(attr)):
        feature_name = attr[i].contents[0].strip()
        feature_value = attr[i].contents[1].get_text().strip()
        feature_value = feature_value.split(' (')[0]
        idx = int(i/4)
        label.append(suffix[idx] + "_" + feature_name)
        record.append(feature_value)
        
#        print(feature_name," - ",feature_value)   
    version = soup.find('a',class_="choose-version").get_text().split(' ')[1]
    return [label,record,version]


def load_dict_nation(filename):
    data = dict()
    with open(filename) as raw_data:
        for item in raw_data:
            if ',' in item:
                key,value = item.split(',', 1)
                data[key]=value.strip()
            else:
                pass # deal with bad lines of text here
    return data


#WC 2010 -> https://sofifa.com/team/111099?v=11&e=156279&set=true
#WC 2014 -> https://sofifa.com/team/111099?v=14&e=157648&set=true
#EUR 2012 -> https://sofifa.com/team/1377?v=12&e=156820&set=true
#EUR 2016 -> https://sofifa.com/team/1377?v=16&e=158375&set=true    
#WC 2018 -> https://sofifa.com/team/1377?v=WC18&e=159107&set=true


#event = {'WC2010':156279,'WC2014':157648}

nation_code = load_dict_nation("nation_code_.csv")

#filename = "2018_FIFA_World_Cup_squads"
filename = "2016_UEFA_Euro_squads"
squad_file = pd.read_csv("squad_member/"+filename+".csv")
nation_list = squad_file['Country'].unique()
nation_list = [st.lower().replace(' ','-') for st in nation_list]

records = []
labels = []
not_found = []
count = 0
for nation in nation_list:
    
    code = nation_code.get(nation,0)
    st = nation + " - " + str(code) + " - "
    print(st,end='')
    
    if code != 0:
        url = "https://sofifa.com/team/"+str(code)+"?v=16&e=158375&set=true"
        label,record,version = retrieve_info(url)                  
        
#        version = ['10','12','14','16','WC18']
        if version == '16':
            print(version+" - ",end='')
            records.append(record)
#            print(len(label))           
#            print(nation,": ",code)
            print("Ok")
            count += 1  
        elif version != None:
            not_found.append(st)
            print(version+" - Wrong Version")
        else:
            not_found.append(st)
            print("Not Found")
    else:
        not_found.append(st)
        print("Not in list")

print(count, '/',len(nation_list))
print(not_found)
labels = label

df = pd.DataFrame.from_records(records,columns=labels)
df.to_csv("squad_strength/"+filename+"_strength__1.csv",index = False)




#count = 0
#for nation in nation_list:
#    
#    code = nation_code.get(nation,0)
#    if code != 0:
#        url = "https://sofifa.com/team/"+str(code)+"?v=11&e=156279&set=true"
#        r = requests.get(url)
##        status = r.status_code
#        html_doc = r.text
#        soup = BeautifulSoup(html_doc,"lxml")
#        version = soup.find('a',class_="choose-version").get_text().split(' ')[1]
#        if version == '11':
#            try:
#                team_name = soup.find("div",class_="info")
#                team_name = team_name.contents[1].get_text().split(' (')
#                team_name = team_name[0].strip()
#                
#                print(nation,": ",team_name)
#                count += 1
#            except:
#                print(nation,": Not Found :",code)
#        else:
#            print(nation,": Not Found Wrong Version :",code)
#            
            
