#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:12:05 2018

@author: chemla
"""


import torch
from .. import utils
from torch import cat
import torch.nn as nn
from collections import OrderedDict
from itertools import accumulate
import numpy as np

import sys
sys.path.append('../..')

from .modules_utils import init_module
from .modules_bottleneck import MLP, DLGMLayer
from .modules_distribution import get_module_from_density


# HiddenModule is an abstraction for inter-layer modules
# it has the following components:
#    hidden_modules : creates hidden modules specified by phidden (can be a list)
#    out_modules : creates the distribution modules specified by platent



class HiddenModule(nn.Module):
    def __init__(self, pins, phidden, pouts=None, linked=True, make_flows=True, *args, **kwargs):
        super(HiddenModule, self).__init__()
        if not issubclass(type(pins), list):
            pins = [pins]
        self.pins = pins
        self.phidden = phidden
        self.pouts = pouts
        
        # get hidden layers
        self.hidden_modules = self.make_hidden_layers(pins, phidden, *args, **kwargs)
        self.linked = linked
        if not linked:
            assert len(phidden) == len(pouts)
        
        # get output layers
        self.out_modules = None
        if pouts:
            self.out_modules = self.make_output_layers(phidden, pouts)
            if make_flows:
                self.flows = self.get_flows(pouts)
                                        
        
    def make_hidden_layers(self,pins, phidden={"dim":800, "nlayers":2, 'label':None, 'conditioning':'concat'}, *args, **kwargs):
        if issubclass(type(phidden), list):
            hidden_modules = [self.make_hidden_layers(pins, ph, *args, **kwargs) for ph in phidden]
        else:
            module_class = phidden.get('class', MLP)
            hidden_modules = module_class(pins, phidden, *args, **kwargs)
        return hidden_modules
            
            
    def make_output_layers(self, pin, pouts):
        '''returns output layers with resepct to the output distribution
        :param in_dim: dimension of input
        :type in_dim: int
        :param pouts: properties of outputs
        :type pouts: dict or [dict]
        :returns: ModuleList'''
        out_modules = []
        if not issubclass(type(pouts), list):
            pouts = [pouts]
        #TODO ça va pas
        for i, pout in enumerate(pouts):
            if issubclass(type(pout),  dict):
                if issubclass(type(pin), list):
                    if self.linked:
                        input_dim = sum([x['dim'] for x in pin])
                        current_encoders = self.hidden_modules
                    else:
                        input_dim = pin[i]['dim']
                        current_encoders = self.hidden_modules[i]
                else:
                    input_dim = pin['dim']
                    current_encoders = self.hidden_modules
                out_modules.append(self.get_module_from_density(pout["dist"])({'dim':input_dim}, pout, encoder=current_encoders))
            else:
                out_modules.append(nn.Linear(input_dim, pout))
        out_modules = nn.ModuleList(out_modules)
        return out_modules
    
    
    def get_module_from_density(self, dist):
        return get_module_from_density(dist)
    
    
    def get_flows(self, pouts):
        flow_modules = []
        if not issubclass(type(pouts), list):
            pouts = [pouts]
        for i, pout in enumerate(pouts):
            flow_parameters = pout.get('flows')
            if flow_parameters:
                FlowModule = flow_parameters.get('class', flows.PlanarFlow)
                n_flows = flow_parameters.get('n_flows', 10)
                current_flow = FlowModule(pout['dim'], n_flows)
                flow_modules.append(current_flow)
            else:
                flow_modules.append(None)
        flow_modules = nn.ModuleList(flow_modules)
        return flow_modules
    
                        
    def forward_hidden(self, x, y=None, *args, **kwargs):
        if issubclass(type(self.hidden_modules), list):
            hidden_out = [h(x, y=y, sample=True, *args, **kwargs) for h in self.hidden_modules]
        else:
            hidden_out = self.hidden_modules(x, y=y, *args, **kwargs)
        
        if self.linked and issubclass(type(self.hidden_modules), list): 
            hidden_out = torch.cat(tuple(hidden_out), 1)
        else:
            hidden_out = hidden_out
        return hidden_out
    
    def forward_params(self, h, y=None, *args, **kwargs):
        z_params = []
        for i, out_module in enumerate(self.out_modules):
            if issubclass(type(self.hidden_modules), nn.ModuleList):
                indices = None
                if out_module.requires_deconv_indices:
                    indices = self.hidden_modules[i].get_pooling_indices()
                if self.linked: 
                    z_params.append(out_module(h, indices=indices))
                else:
                    z_params.append(out_module(h[i], indices=indices))
            else: 
                indices = None; output_size = None
                if out_module.requires_deconv_indices:
                    indices = self.hidden_modules.get_pooling_indices()
                    output_size = self.hidden_modules.encoder.get_output_conv_length()[1][0]
                z_params.append(out_module(h, indices=indices,output_size=output_size))
                    
        if not issubclass(type(self.pouts), list):
            z_params = z_params[0]
        return z_params
        
    def sample(self, z_params, sample=True):
        if issubclass(type(self.pouts), list):
            z = list()
            for i in range(len(z_params)):
                if sample:
                    try:
                        current_z = self.pouts[i]['dist'](*z_params[i]).rsample() 
                    except NotImplementedError:
                        current_z = self.pouts[i]['dist'](*z_params[i]).sample() 
                else:
                    current_z =self.pouts[i]['dist'](*z_params[i]).mean 
                if not self.pouts[i].get('flows') is None:
                    current_z = self.flows[i](current_z)
                z.append(current_z)
        else:
            if sample:
                try:
                    z = self.pouts['dist'](*z_params).rsample()
                except NotImplementedError:
                    z = self.pouts['dist'](*z_params).sample()
                    pass
            else:
                z = self.pouts['dist'](*z_params).mean
            if not self.pouts.get('flows') is None and hasattr(self, 'flows'):
                z = self.flows[0](z)
        return z

    def forward(self, x, y=None, sample=True, *args, **kwargs):
        out = {}
        out['hidden'] = self.forward_hidden(x, y=y, *args, **kwargs)
        if self.out_modules!=None:
            out['out_params'] = self.forward_params(out['hidden'], y=y, *args, **kwargs)
            if sample:
                out['out'] = self.sample(out['out_params'])
        return out




class DLGMModule(HiddenModule):
    ''' Specific decoding module for Deep Latent Gaussian Models'''
    def get_module_from_density(self, dist):
        return DLGMLayer
    
    def forward_params(self, h, eps, *args, **kwargs):
        z_params = []
        for i, out_module in enumerate(self.out_modules):
            if issubclass(type(self.hidden_modules), nn.ModuleList):
                if self.linked: 
                    z_params.append(out_module(h, eps))
                else:
                    z_params.append(out_module(h[i], eps[i]))
            else: 
                z_params.append(out_module(h, eps))
        if not issubclass(type(self.pouts), list):
            z_params = z_params[0]
        return z_params
    
    def sample(self, z_params, sample=True, *args, **kwargs):
        if issubclass(type(self.pouts), list):
            z = []
            for i in range(len(z_params)):
                z.append(z_params[0]+z_params[1])
        else:
            z = z_params[0] + z_params[1]
        return z

