# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 23:25:23 2018

@author: mrthl
"""
import pandas as pd
import glob, os
import sys
from unidecode import unidecode

#import 
filename = sys.argv[1]

#filename = "fifa08_4"

df_list = []
for file in glob.glob(filename+"_*.csv"):
    print(file)
#    df = pd.read_csv(file)
    df = pd.read_csv(file,encoding='iso-8859-1')
    df_list.append(df)

df_concat = pd.concat(df_list)
df_concat = df_concat.drop_duplicates()

# Check number of row deleted
sum = 0
for i in range(len(df_list)):
    data_num = df_list[i].shape[0]
    sum += data_num
    print("Shape of data {}: {}".format(i,data_num))

print("Total number of data row: ",sum)
print("Number of deleted row: ",sum - df_concat.shape[0])

df_concat.to_csv(filename+".csv",encoding='utf-8',index_label = False)

#df_load = pd.read_csv(filename+".csv")


#df1 = pd.read_csv("fifa08_4_1.csv")
#df2 = pd.read_csv("fifa08_4_2.csv")
#
#df_concat = pd.concat([df1,df2],join='inner',keys='Name')
#df_concat = df_concat.drop_duplicates()
#df1.shape[0] + df2.shape[0] 
#
#a = pd.read_csv("fifa09_5_1.csv",encoding='iso-8859-1')

#os.chdir("/mydir")

    
    