# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 11:31:51 2018

@author: mrthl
"""
import pandas as pd
import gc

# User Define Class
from utilities.helper_function import loadData, MLmodel
#import pickle

# Define a program parameter
exp_name = '1'
encode_name = 'result_ex'+exp_name
filename = "data/data_ex"+exp_name+".csv"

# Read data
data = pd.read_csv(filename, encoding='utf-8')

#data_x, data_y, x_train, x_test, y_train, y_test = loadData(data)

#data_x, data_y, x_train, x_test, y_train, y_test = loadData(data,scaler=False,home_team=False,encode_name = encode_name,synthesize=False)
#list_data = [data_x, data_y, x_train, x_test, y_train, y_test]

model_list = ['DT_1','DT_2','LR','RF','GBT','ADA','NN','LGBM']
#model_list = ['DT_1','DT_2']
record = []

for model_name in model_list:
    data_x, data_y, x_train, x_test, y_train, y_test = loadData(data,scaler=False,home_team=False,encode_name = encode_name,synthesize=False)
    if (model_name == 'DT_1'):
        data_x = data_x.loc[:,"odd_diff_win":"odd_draw"]
        x_train = x_train.loc[:,"odd_diff_win":"odd_draw"]
        x_test = x_test.loc[:,"odd_diff_win":"odd_draw"]
    elif (model_name == 'DT_2'):
        data_x = data_x.loc[:,"h2h_win_diff":"form_diff_draw"]
        x_train = x_train.loc[:,"h2h_win_diff":"form_diff_draw"]
        x_test = x_test.loc[:,"h2h_win_diff":"form_diff_draw"]
    
    list_data = [data_x, data_y, x_train, x_test, y_train, y_test]
    
    model = MLmodel(model_name,list_data,encode_name,'1')
    record.append([model_name,model['CVerror'],model['f1'],model['auroc']])
    del model
    gc.collect()


#df['CV error'] = df['CV error'].apply(shit)

df = pd.DataFrame.from_records(record,columns = ["Name","CV error","F1 Micro","Area ROC"])
df = df.set_index(['Name'])
df.plot(kind='bar',subplots=True,title="")

#    record.append([])
#    {'model':modelCV,'auroc':auroc_micro,'f1':f1_score_micro,'CVerror':scores}    

#model = MLmodel(model_list[0],list_data,encode_name)


