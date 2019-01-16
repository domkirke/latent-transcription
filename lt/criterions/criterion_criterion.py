#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:26:17 2018

@author: chemla
"""
import torch
import torch.nn as nn
import copy, pdb
import numpy as np
from ..utils import decudify

class Criterion(nn.Module):
    def __init__(self, options={}, weight=1.0):
        super(Criterion, self).__init__()
        self.weight = weight
        self.loss_history = {}
        
    def loss(self, *args, **kwargs):
        return 0.
    
    def step(self):
        return 0.
    
    def get_named_losses(self, losses):
        return {'dummy': losses}

    def write(self, name, losses):
        losses = self.get_named_losses(losses)
        if not name in self.loss_history.keys():
            self.loss_history[name] = {}
        for loss_name, value in losses.items():
            if not loss_name in list(self.loss_history[name].keys()):
                self.loss_history[name][loss_name] = []
            value = decudify(value, scalar=True)
            '''
            if issubclass(type(value), list) or issubclass(type(value), tuple):
                value  = [x.cpu() for x in value]
            else:
                value = value.cpu()
            '''
            self.loss_history[name][loss_name].append(value)

    
    def __call__(self, *args, **kwargs):
        l, losses = self.loss(*args, **kwargs)
        return self.weight * l, losses
        
    def __add__(self, c):
        if issubclass(type(c), LossContainer):
            nc = copy.deepcopy(c)
            nc.criterions_.append(self)
        if issubclass(type(c), Criterion):
            c = copy.deepcopy(c)
            nc = LossContainer([self, c])
        return nc
    
    def __radd__(self, c):
        return self.__add__(c)
        
    def __sub__(self, c):
        if issubclass(type(c), LossContainer):
            nc = copy.deepcopy(c)
            c.weight = -1.0
            nc.criterions_.append(self)
        if issubclass(type(c), Criterion):
            c = copy.deepcopy(c)
            c.weight = -1.0
            nc = LossContainer([self, c])
        return nc
    
    def __rsub__(self, c):
        return self.__sub__(c)
        
    def __mul__(self, f):
        assert issubclass(type(f), float) or issubclass(type(f), np.ndarray)
        new = copy.deepcopy(self)
        new.weight *= f
        return new
        
    def __rmul__(self, c):
        return self.__mul__(c)
 
    def __div__(self, f):
        assert issubclass(type(f), float)
        new = copy.deepcopy(self)
        new.weight /= f
        return new
        
    def __rdiv__(self, c):
        return self.__div__(c)
        
     
class LossContainer(Criterion):
    def __init__(self, criterions=[], options={}, weight=1.0):
        super(Criterion, self).__init__()
        self.criterions_ = criterions
    
    def loss(self, *args, **kwargs):
        full_losses = [c(*args, **kwargs) for c in self.criterions_]
        loss = 0.; losses = []
        for l, ls in full_losses:
            loss = loss + l
            losses.append(ls)
        return loss, losses
    
    def step(self):
        for i, l in enumerate(self.criterions_):
            l.step()
    
    def get_named_losses(self, losses):
        named_losses=dict()
        for i, l in enumerate(losses):
            current_loss = self.criterions_[i].get_named_losses(l)
            named_losses = {**named_losses, **current_loss}
        return named_losses

            
            

