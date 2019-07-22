# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:59:56 2018

@author: hp
"""
import numpy as np

def accuracy(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true), ', set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def precision(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        elif len(set_pred) == 0:
            tmp_a = 0
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float(len(set_pred))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def recall(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float(len(set_true))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def f_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = (2*len(set_true.intersection(set_pred)))/\
                    float( len(set_true) + len(set_pred))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def cls_recall(y_true, y_pred, normalize=True, sample_weight=None):
    t_count = 0
    count = 0
    for i in range(y_true.shape[0]):
        if(y_true[i] == 1):
            t_count = t_count + 1
            if(y_true[i] == y_pred[i]):
                count = count + 1
    return(count/t_count)
    
def cls_precision(y_true, y_pred, normalize=True, sample_weight=None):
    t_count = 0
    count = 0
    for i in range(y_true.shape[0]):
        if(y_pred[i] == 1):
            t_count = t_count + 1
            if(y_true[i] == y_pred[i]):
                count = count + 1
    return(count/t_count)