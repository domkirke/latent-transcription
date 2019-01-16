#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 23:27:11 2018

@author: chemla
"""
import torch.distributions as dist

class Prior(object):
    def __init__(self, dim=0, *args, **kwargs):
#        super(Prior, self).__init__()
        self.dim = dim
        self.dist = dist.Distribution
        self.params = ()
        
    def __call__(self, x, cuda=False, device=None, *args, **kwargs):
        draw = self.dist.rsample(*self.params)
        return draw
    
    def get_params(self, device='cpu', *args, **kwargs):
        params = [ p.to(device) for p in self.params ]
        return tuple(params)

