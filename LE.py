# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:09:29 2018

@author: mrthl
"""

from sklearn.preprocessing import LabelEncoder
import numpy as np

def saveLabelEncoder(x,file):
    encoder = LabelEncoder()    
    encoder.fit(x)
    np.save(file, encoder.classes_)
    return encoder

def loadLabelEncoder(file):
    encoder = LabelEncoder()
    encoder.classes_ = np.load(file)