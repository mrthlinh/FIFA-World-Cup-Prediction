# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:31:42 2018

@author: mrthl
"""
import pandas as pd
import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score
from utilities.helper_function import loadLabelEncoder
# User Define Class
#from utilities.helper_function import MyReport, loadData
import pickle

# Load data
data = pd.read_csv("data/data_ex3.csv", encoding='utf-8')
data_x = data.iloc[:,6:-2]
data_y = data.iloc[:,-2]
data_y_wdl = data.iloc[:,-1]
encoder = loadLabelEncoder("LE/result_ex2.npy")
data_y = encoder.transform(data_y)

model_list = ['DT_1','DT_2','DT_3','LR','RF','GBT','ADA','NN','LGBM']

record = []
for model_name in model_list:
    file_model = 'save_model/'+model_name+'_2.sav'
    model = pickle.load(open(file_model,'rb'))
    print(model.get_params)
    
    
    if (model_name == 'DT_1'):
        y_pred = model.predict_proba(data_x.loc[:,"odd_diff_win":"odd_draw"])
    elif (model_name == 'DT_2'):
        y_pred = model.predict_proba(data_x.loc[:,"h2h_win_diff":"form_diff_draw"])
    elif (model_name == 'DT_3'):
        y_pred = model.predict_proba(data_x.loc[:,"game_diff_ovr":"game_diff_def_teamwidth"])        
    else:
        y_pred = model.predict_proba(data_x)
        
    def topK(row,k):
        top = pd.Series.sort_values(row)
        return top.index[-k]
    
    y_pred = pd.DataFrame(y_pred)
    
    top1 = y_pred.apply(topK,k=1, axis = 1)
    top1_acc = 100*accuracy_score(top1,data_y)
    print("Top 1 Accuracy: ",top1_acc)
    
    f1_micro = 100*f1_score(data_y,top1,average='micro')
    print("f1_micro: ",f1_micro)
    
#    area = roc_auc_score(data_y,top1,average='micro')
#    print("area: ",area)
    # W / D / L acc
    wdl_pred = pd.DataFrame()    
    wdl_pred['win'] = y_pred.apply(lambda row: np.sum(row[4:]),axis=1)
    wdl_pred['draw'] = y_pred.apply(lambda row: row[0],axis=1)
    wdl_pred['lose'] = y_pred.apply(lambda row: np.sum(row[1:4]),axis=1)
    
    top1_wdl = wdl_pred.apply(topK,k=1, axis = 1)
    top1_wdl_acc = 100*accuracy_score(data_y_wdl,top1_wdl)
    print("Win/Draw/Lose accuracy: ",top1_wdl_acc)
    
    record.append([model_name,top1_acc,top1_wdl_acc,f1_micro])
    
    
df = pd.DataFrame.from_records(record,columns = ["Name","Goal Difference Acc","W-D-L Acc","F1 Micro"])
df = df.set_index(['Name'])
df.plot(kind='bar',subplots=True,title="")



#axes = df.plot.bar(rot=0, subplots=True)
#>>> axes[1].legend(loc=2)  


#def visualization(data,kind,title,figsize):
#    ax = data.plot(kind=kind,title = title,figsize=figsize)
#    shape = data.shape[0]
#    # set individual bar lables using above list
#    patches = ax.patches
#    for i in range(len(patches)):
#        patch = patches[i]
#        x = patch.get_width() + 0.05
#        y = patch.get_y() + 0.05
#        
#        
#        score = data.iloc[i % shape,int(i/shape)]
##        print(i,'--',score)
#    
#        # get_width pulls left or right; get_y pushes up or down
#        ax.text(x, y, str(score), fontsize=10,color='black')

#visualization(df["Goal Difference Acc"],'barh',"Goal Difference",(10,5))
#pd.DataFrame.from_records(records,columns=columns)