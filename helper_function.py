# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:34:25 2018

@author: mrthl
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import itertools
import matplotlib.pyplot as plt

from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix

def convert_moneyline_decimal(st):
    '''
    Description: 
    Input:

    Output:
        
    '''
    try:
        num = float(st)
    except:
        return None
    result = 0
    if num > 0:
        result = num / 100 + 1
    elif num < 0:
        result = -100 / num + 1
    return round(result,2)


def hist_feature(row,df_hist,bool_diff):     
    '''
    Description: Lookup "head-to-head" feature - difference of wins and number of draw
    Input:
        row: Pandas Series with name "team_1" and "team_2"
        df_hist: Pandas Dataframe, all matches (database)
    Output:
        [diff_win,num_draw]
    '''
#    team1= correct_Korea(row.team_1)
#    team2= correct_Korea(row.team_2)
    team1= row.team_1
    team2= row.team_2  
    df_1 = df_hist.loc[(df_hist['home_team'] == team1) & (df_hist['away_team'] == team2),'home_score':'away_score']     
    df_2 = df_hist.loc[(df_hist['home_team'] == team2) & (df_hist['away_team'] == team1),'home_score':'away_score']
    df_2.columns = ['away_score','home_score']
    df_3 = pd.concat([df_1,df_2])
    diff = df_3['home_score'] - df_3['away_score']
    result = diff.apply(w_d_l)
    result = result.value_counts()
    
    num_win = result.get('win',0)
    num_draw = result.get('draw',0)
    num_lose = result.get('lose',0)
    
    diff_win = num_win - num_lose
    
#    diff = True
    if bool_diff:
        return [diff_win,num_draw]
    else:    
        return [num_win,num_lose,num_draw]


def form_feature(row,df_hist,num_recent_matches,bool_diff):
    '''
    Description: Lookup "form" feature - "goal for", "goal against", "win", "draw"
    Input:
        row: Pandas Series with name "team_1" and "team_2"
        df_hist: Pandas Dataframe, all matches (database)
    Output:
        [team1_info['f_goalF'],team2_info['f_goalF'],team1_info['f_goalA'],team2_info['f_goalA'],
            team1_info['f_win'],team2_info['f_win'],team1_info['f_draw'],team2_info['f_draw']]
    '''    
    
    def form(team,num_recent_matches):
        '''
        Description:
        Input:
            
        Output:
            
        '''
    
        df = df_hist.loc[(df_hist['home_team'] == team) | (df_hist['away_team'] == team),'date':'away_score']
        df = df.tail(num_recent_matches)
        
        df_home = df.loc[df['home_team'] == team,'home_score':'away_score']
        df_away = df.loc[df['away_team'] == team,'home_score':'away_score']
        df_away.columns = ['away_score','home_score']
        df_sum = pd.concat([df_home,df_away])    
        goal = df_sum.apply(np.sum,axis=0) 
        
        diff = df_sum['home_score'] - df_sum['away_score']
        result = diff.apply(w_d_l)
        result = result.value_counts()
        
        f_goalF = goal['home_score']
        f_goalA = goal['away_score']
        f_win   = result.get('win',0)
        f_draw   = result.get('draw',0)
        
        return {"f_goalF":f_goalF,"f_goalA":f_goalA,
                "f_win":f_win, "f_draw":f_draw }

    
#    team1= correct_Korea(row.team_1)
#    team2= correct_Korea(row.team_2)
    team1= row.team_1
    team2= row.team_2 
#    num_recent_matches = 10
    
    team1_info = form(team1,num_recent_matches)
    team2_info = form(team2,num_recent_matches)
    
    if bool_diff:    
        return [team1_info['f_goalF'] - team2_info['f_goalF'],team1_info['f_goalA'] - team2_info['f_goalA'],
                team1_info['f_win'] - team2_info['f_win'],team1_info['f_draw'] - team2_info['f_draw']]
    else:
        return [team1_info['f_goalF'],team2_info['f_goalF'],team1_info['f_goalA'],team2_info['f_goalA'],
                team1_info['f_win'],team2_info['f_win'],team1_info['f_draw'],team2_info['f_draw']]

def hist_feature_new(row,df_hist,bool_diff):     
    '''
    Description: Lookup "head-to-head" feature - difference of wins and number of draw
    Input:
        row: Pandas Series with name "team_1" and "team_2"
        df_hist: Pandas Dataframe, all matches (database)
    Output:
        [diff_win,num_draw]
    '''
#    team1= correct_Korea(row.team_1)
#    team2= correct_Korea(row.team_2)
    team1= row.team_1
    team2= row.team_2  
    df_1 = df_hist.loc[(df_hist['team_1'] == team1) & (df_hist['team_2'] == team2),['score_1','score_2']]     
    df_2 = df_hist.loc[(df_hist['team_1'] == team2) & (df_hist['team_2'] == team1),['score_1','score_2']]
    df_2.columns = ['score_2','score_1']
    df_3 = pd.concat([df_1,df_2])
    diff = df_3['score_1'] - df_3['score_2']
    result = diff.apply(w_d_l)
    result = result.value_counts()
    
    num_win = result.get('win',0)
    num_draw = result.get('draw',0)
    num_lose = result.get('lose',0)
    
    diff_win = num_win - num_lose
    
#    diff = True
    if bool_diff:
        return [diff_win,num_draw]
    else:    
        return [num_win,num_lose,num_draw]

def form(team,df_hist,num_recent_matches):
        '''
        Description:
        Input:
            
        Output:
            
        '''
    
        df = df_hist.loc[(df_hist['team_1'] == team) | (df_hist['team_2'] == team),'match_date':'score_2']
        df = df.tail(num_recent_matches)
        
        df_team1 = df.loc[df['team_1'] == team,['score_1','score_2']]
        df_team2 = df.loc[df['team_2'] == team,['score_1','score_2']]
        df_team2.columns = ['score_2','score_1']
        df_sum = pd.concat([df_team1,df_team2])    
        goal = df_sum.apply(np.sum,axis=0) 
        
        diff = df_sum['score_1'] - df_sum['score_2']
        result = diff.apply(w_d_l)
        result = result.value_counts()
        
        f_goalF = goal['score_1']
        f_goalA = goal['score_2']
        f_win   = result.get('win',0)
        f_draw   = result.get('draw',0)
        
        return {"f_goalF":f_goalF,"f_goalA":f_goalA,
                "f_win":f_win, "f_draw":f_draw }
        
def form_feature_new(row,df_hist,num_recent_matches,bool_diff):
    '''
    Description: Lookup "form" feature - "goal for", "goal against", "win", "draw"
    Input:
        row: Pandas Series with name "team_1" and "team_2"
        df_hist: Pandas Dataframe, all matches (database)
    Output:
        [team1_info['f_goalF'],team2_info['f_goalF'],team1_info['f_goalA'],team2_info['f_goalA'],
            team1_info['f_win'],team2_info['f_win'],team1_info['f_draw'],team2_info['f_draw']]
    '''    
    
    def form(team,num_recent_matches):
        '''
        Description:
        Input:
            
        Output:
            
        '''
    
        df = df_hist.loc[(df_hist['team_1'] == team) | (df_hist['team_2'] == team),'match_date':'score_2']
        df = df.tail(num_recent_matches)
        
        df_team1 = df.loc[df['team_1'] == team,['score_1','score_2']]
        df_team2 = df.loc[df['team_2'] == team,['score_1','score_2']]
        df_team2.columns = ['score_2','score_1']
        df_sum = pd.concat([df_team1,df_team2])    
        goal = df_sum.apply(np.sum,axis=0) 
        
        diff = df_sum['score_1'] - df_sum['score_2']
        result = diff.apply(w_d_l)
        result = result.value_counts()
        
        f_goalF = goal['score_1']
        f_goalA = goal['score_2']
        f_win   = result.get('win',0)
        f_draw   = result.get('draw',0)
        
        return {"f_goalF":f_goalF,"f_goalA":f_goalA,
                "f_win":f_win, "f_draw":f_draw }

    
#    team1= correct_Korea(row.team_1)
#    team2= correct_Korea(row.team_2)
    team1= row.team_1
    team2= row.team_2 
#    num_recent_matches = 10
    
    team1_info = form(team1,num_recent_matches)
    team2_info = form(team2,num_recent_matches)
    
    if bool_diff:    
        return [team1_info['f_goalF'] - team2_info['f_goalF'],team1_info['f_goalA'] - team2_info['f_goalA'],
                team1_info['f_win'] - team2_info['f_win'],team1_info['f_draw'] - team2_info['f_draw']]
    else:
        return [team1_info['f_goalF'],team2_info['f_goalF'],team1_info['f_goalA'],team2_info['f_goalA'],
                team1_info['f_win'],team2_info['f_win'],team1_info['f_draw'],team2_info['f_draw']]
  
#def addFeature_diff(new_features,old_features,my_df,drop=True):  
#    df = my_df.copy()
#    for i in range(len(new_features)-1):
#        new_feature = new_features[i]
#        old_feature_1 = old_features[i] + "1"
#        old_feature_2 = old_features[i] + "2"
#        
#        df[new_feature] = df.apply(lambda row: (row[old_feature_1] - row[old_feature_2]),axis=1)
#        if (drop):
#            df = df.drop([old_feature_1,old_feature_2],axis=1)
##    'avg_odds_draw' -> no need to take difference
#    df[new_features[-1]] = my_df[old_features[-1]]
#    if (drop):
#        df = df.drop([old_features[-1]],axis=1)
#    return df



def w_d_l(goal_diff):
    '''
    Description: Derive a label Win/Draw/Lose based on goal difference
    Input:
        goal_diff: int
    Output:
        label: String
    '''
    if goal_diff > 0:
        return 'win'
    elif goal_diff == 0:
        return 'draw'
    else:
        return 'lose'
    
def goalDiff_label(diff):
    '''
    Description: This function turns the goal differences into labels based on following rules
    Input: 
        goal differences:   int
    Output: 
        labels: string
    
        Team A vs Team B
        diff = 0  -> draw_0 (A draws B)
        diff = 1  -> win_1 (A wins B with 1 goal difference)
        diff = 2  -> win_2 (A wins B with 2 goal differences)
        diff >= 3 -> win_3 (A wins B with 3 or more goal differences)
        diff = -1 -> lose_1 (B wins A with 1 goal difference)
        diff = -2 -> lose_2 (B wins A with 2 goal differences)
        diff <= -3-> lose_3 (B wins A with 3 or more than 3 goal differences)
    
    '''
    
    if diff == 0:
        return 'draw_0'
    elif diff == 1:
        return 'win_1'
    elif diff == 2:
        return 'win_2'
    elif diff >= 3:
        return 'win_3'
    elif diff == -1:
        return 'lose_1'
    elif diff == -2:
        return 'lose_2'
    elif diff <= -3:
        return 'lose_3'

def reverse_result(diff):
    '''
    Description: This function reverse the match results for data systhesis
    Input:  
        labels: string
    Output: 
        labels: string 
            win -> lose and lose -> win
    
    '''
    
    if diff == 'draw_0':
        return 'draw_0'
    elif diff == 'win_1':
        return 'lose_1'
    elif diff == 'win_2':
        return 'lose_2'
    elif diff == 'win_3':
        return 'lose_3'
    elif diff == 'lose_1':
        return 'win_1'
    elif diff == 'lose_2':
        return 'win_2'
    elif diff == 'lose_3':
        return 'win_3'

def result_proba(result):
    '''
    Description: This function converts simulation results to probability
    Input:  
        result : Dataframe, for N simulation time
    Output: 
        probability: Dataframe 
    
    '''
     
    result_count = pd.DataFrame()
    result_count['win_1'] = result.apply(lambda x: x.value_counts().get('win_1',0),axis=1)
    result_count['win_2'] = result.apply(lambda x: x.value_counts().get('win_2',0),axis=1)
    result_count['win_3'] = result.apply(lambda x: x.value_counts().get('win_3',0) ,axis=1)
    result_count['lose_1'] = result.apply(lambda x: x.value_counts().get('lose_1',0) ,axis=1)
    result_count['lose_2'] = result.apply(lambda x: x.value_counts().get('lose_2',0),axis=1)
    result_count['lose_3'] = result.apply(lambda x: x.value_counts().get('lose_3',0),axis=1)
    result_count['draw_0'] = result.apply(lambda x: x.value_counts().get('draw_0',0),axis=1)
    
    result_prob = pd.DataFrame()
    result_prob['win_1'] = result_count.apply(lambda x: 100*x[0] / np.sum(x), axis = 1)
    result_prob['win_2'] = result_count.apply(lambda x: 100*x[1] / np.sum(x), axis = 1)
    result_prob['win_3'] = result_count.apply(lambda x: 100*x[2] / np.sum(x), axis = 1)
    result_prob['lose_1'] = result_count.apply(lambda x: 100*x[3] / np.sum(x), axis = 1)
    result_prob['lose_2'] = result_count.apply(lambda x: 100*x[4] / np.sum(x), axis = 1)
    result_prob['lose_3'] = result_count.apply(lambda x: 100*x[5] / np.sum(x), axis = 1)
    result_prob['draw_0'] = result_count.apply(lambda x: 100*x[6] / np.sum(x), axis = 1)
    
    return result_prob

def addFeature_diff(new_features,old_features,my_df,drop=True):  
    '''
    Description: This function replaces defined "old feature" with defined "new feature" 
    
    Input:  
        my_df:  Dataframe
        old_features: Array Str
        new_features: Array Str
        
    Output: 
        Probability: Dataframe
    '''
    
    df = my_df.copy()
    for i in range(len(new_features)-1):
        new_feature = new_features[i]
        old_feature_1 = old_features[i] + "1"
        old_feature_2 = old_features[i] + "2"
        
        df[new_feature] = df.apply(lambda row: (row[old_feature_1] - row[old_feature_2]),axis=1)
        if (drop):
            df = df.drop([old_feature_1,old_feature_2],axis=1)
#    'avg_odds_draw' -> no need to take difference
    df[new_features[-1]] = my_df[old_features[-1]]
    if (drop):
        df = df.drop([old_features[-1]],axis=1)
    return df


def saveLabelEncoder(x,file):
    '''
    Description: Encode label and save model to "file" 
    
    Input:
        x:  Pandas Series
        file:   String
    Output:
        Encoder
    '''
    
    encoder = LabelEncoder()    
    encoder.fit(x)
    np.save(file, encoder.classes_)
    return encoder

def loadLabelEncoder(file):
    '''
    Description: Load encode label
    
    Input:
        file: String
    Output:
        Encoder
        
    '''
    
    encoder = LabelEncoder()
    encoder.classes_ = np.load(file)
    return encoder

def loadData(data, scaler = True, home_team = True,test_size=0.3):
    '''
    Description: load data and standardize data
    
    Input:
        data:   Dataframe
        scaler: Boolean, optional (default = True), standardize data
        home_team: Boolean, optional (default = True), decode "home_team" or not
        test_size: ratio for dataset division
    Output:
        [x,y,x_train, x_test, y_train, y_test]
        x: features
        y: label
        
    '''
    
    data_ = data.iloc[:,2:]
    x = data_.iloc[:,:-1]    
    y = data_.iloc[:,-1]
    
    # Label Encoder 'Result'
    encoder = LabelEncoder()    
    y = encoder.fit_transform(y)
    
    if home_team:
        same_ht = x.team_1 == x.home_team
        x.loc[same_ht,'home_team'] = 1
        x.loc[-same_ht,'home_team'] = 0
    else:
        x = x.drop(columns=['home_team'])
    x = x.drop(columns=['team_1','team_2','tournament'])
    
    if scaler:
        x.iloc[:,1:] = StandardScaler().fit_transform(x.iloc[:,1:])
        
    x_train, x_test, y_train, y_test = train_test_split(x,y.squeeze(),test_size=test_size, random_state=85)
    
    return [x,y,x_train, x_test, y_train, y_test]

   
def MyReport(model,model_Name,list_data,tune = True):
    """
    Need to write sth here
    """
    
    data_x, data_y, x_train, x_test, y_train, y_test = list_data
    # Training
    model.fit(x_train,y_train)
    
    modelCV = model
    if tune:                             
        modelCV = model.best_estimator_

    # General Report          
    y_predCV = modelCV.predict(x_test)
    print(classification_report(y_test, y_predCV))
     
    # Plot Confusion Matrix
    le_result = loadLabelEncoder('LE/result.npy')
    class_names = le_result.classes_
    cnf_matrix = confusion_matrix(y_test, y_predCV)
    plot_confusion_matrix(cnf_matrix, classes=class_names,title=model_Name+' Confusion matrix, without normalization')
    
    # ROC curve
    try:
        y_score = modelCV.decision_function(x_test)
        plot_ROC_curve(y_test,y_score,title=model_Name+' ROC curve',class_names = class_names)
    except:
        print("ROC curve is not available because model does not have decision_function method")
    # 10-fold-test error
    scores = cross_val_score(modelCV, data_x, data_y, cv=10)
    print("10-fold cross validation mean square error: ",np.mean(scores))
    return modelCV

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_ROC_curve(y_test,y_score,title,class_names):
    """
    Need to write sth here
    """
    
    # Binarize the output
#    y = label_binarize(y_test, classes=[0, 1, 2])
    y = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y.shape[1]
    
    # Compute ROC curve and ROC area for each class
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

#    for i, color in zip(class_names, colors):
#        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                 label='ROC curve of class {0} (area = {1:0.2f})'
#                 ''.format(i, roc_auc[i]))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class "{0}" (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

