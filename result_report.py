# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:12:59 2018

@author: mrthl
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np

from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix

from LE import loadLabelEncoder

def MyReport(model,model_Name,list_data,tune = True):
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