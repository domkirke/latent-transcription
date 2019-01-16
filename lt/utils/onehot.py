#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:46:21 2018

@author: chemla
"""
import pdb
from numbers import Number
import torch, numpy as np
from torch.autograd import Variable

def oneHot(labels, dim):
    if isinstance(labels, Number):
        t = np.zeros((1, dim))
        t[0, int(labels)] = 1
    elif issubclass(type(labels), np.ndarray):
        n = labels.shape[0]
        t = np.zeros((n, dim))
        for i in range(n):
            t[i, int(labels[i])] = 1
    elif issubclass(type(labels), Variable):
        n = labels.size(0)
        t = torch.Tensor(n, dim).zero_()
        for i in range(n):
            t[i, int(labels[i])] = 1
    elif issubclass(type(labels), torch.Tensor):
        n = labels.size(0)
        t = torch.Tensor(n, dim).zero_()
        for i in range(n):
            t[i, int(labels[i])] = 1
    else:
        raise Exception('type %s is not recognized by oneHot function'%type(labels))        
    return t

def fromOneHot(vector):
    if issubclass(type(vector), np.ndarray):
        ids = np.argmax(vector, axis=1)
        return ids[1]
    if issubclass(type(vector), torch.Tensor):
        return torch.argmax(vector, dim=1)
    return ids[1]