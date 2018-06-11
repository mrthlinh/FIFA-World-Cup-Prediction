# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:36:48 2018

@author: mrthl
"""
import numpy as np
import pandas as pd
from unidecode import unidecode

dire = "E:/UTD/Spring Board Data Science/Capstone/FIFA World Cup Prediction/web_crawler/player/"
filename = "fifa18"
# filename = "fifa17_173"
#filename = "fifa08_4"
#filename = "fifa07_3"
#filename = "fifa06_2"
#filename = "fifa05_1"
df = pd.read_csv(dire+ filename+".csv",encoding='iso-8859-1')
#version = "beta_clean"
col = df.columns.tolist()

temp_col = []

Club = []
Nation = []
BallSkills = []
Defence =[]
Mental = []
Passing = []
Physical = []
Shooting = []
Goalkeeper = []

def findElement(name,list_in,list_out,startswith=False):
    for element in list_in:
        if (startswith):
            if element.startswith(name):
                list_out.append(element)
        else:
            if element == name:
                list_out.append(element)


order = ['Name','Country','Club','OverRate','PotRate','Height','Weight','PreferredFoot',
         'BirthDate','Age','PreferredPositions']


for i in range(len(order)):
    findElement(order[i],col,temp_col,startswith=False)


findElement("Club_",col,Club,startswith=True)
findElement("Nation_",col,Nation,startswith=True)
findElement("BallSkills",col,BallSkills,startswith=True)
findElement("Defence",col,Defence,startswith=True)
findElement("Mental",col,Mental,startswith=True)
findElement("Passing",col,Passing,startswith=True)
findElement("Physical",col,Physical,startswith=True)
findElement("Shooting",col,Shooting,startswith=True)
findElement("Goalkeeper",col,Goalkeeper,startswith=True)

#new_col.append(Nation)

new_col = temp_col + Nation + Club + BallSkills + Defence + Mental + Passing + Physical + Shooting + Goalkeeper

new_df = df[new_col]

# Change all name to English
new_df['Name'] = new_df['Name'].apply(unidecode)
new_df['Country'] = new_df['Country'].apply(unidecode)
new_df['Club'] = new_df['Club'].apply(lambda row: unidecode(row) if type(row) != float else row )


new_df['Club_ContractLength'] = np.where(new_df.Club_ContractLength.isnull(),new_df.Nation_ContractLength,new_df.Club_ContractLength)

new_df = new_df.drop(['Nation_ContractLength'], axis = 'columns')
# Write a test to check missing all value of Nation
na_NationPos= np.count_nonzero(new_df['Nation_Position'].isnull())
na_NationKitNum= np.count_nonzero(new_df['Nation_KitNumber'].isnull())
print("Checking for players not playing for Nation: ",na_NationPos == na_NationPos)

# Write a test to check missing all value of Nation


#new_df['Club_ContractLength'] = new_df[['Club_ContractLength','Nation_ContractLength']]
#a = new_df[['Club_ContractLength','Nation_ContractLength']]
#a['result'] = new_df[['Club_ContractLength','Nation_ContractLength']].apply(lambda row: row[0] if row[0]., axis = 'columns')
#a['result'] = np.where(a.Club_ContractLength.isnull(),a.Nation_ContractLength,a.Club_ContractLength)

new_df.to_csv(filename+'_clean.csv', index = False,encoding='utf-8')
