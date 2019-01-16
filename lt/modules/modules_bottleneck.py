#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:49:58 2017

@author: chemla
"""
import torch
from .. import utils
from torch import cat
import torch.nn as nn
from collections import OrderedDict
import numpy as np

import sys
sys.path.append('../..')

from .modules_utils import init_module
# Full modules for variational algorithms


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, nn_lin="ReLU", batch_norm='batch', dropout=None, name_suffix="", *args, **kwargs):
        super(MLPLayer, self).__init__()
        self.input_dim = input_dim; self.output_dim = output_dim
        self.name_suffix = name_suffix
        self.batch_norm = batch_norm 
        modules = OrderedDict()
        modules["hidden"+name_suffix] =  nn.Linear(input_dim, output_dim)
        init_module(modules["hidden"+name_suffix], nn_lin)
        if batch_norm:
            if batch_norm == 'batch':
                modules["batch_norm_"+name_suffix]= nn.BatchNorm1d(output_dim)
            if batch_norm == 'instance':
                modules["instance_norm_"+name_suffix]= nn.InstanceNorm1d(1)
        if not dropout is None:
            modules['dropout_'+name_suffix] = nn.Dropout(dropout)
        self.nn_lin = nn_lin
        if nn_lin:
            modules["nnlin"+name_suffix] = getattr(nn, nn_lin)()
        self.module = nn.Sequential(modules)
        
    def forward(self, x):
        out = self.module._modules['hidden'+self.name_suffix](x)
        if self.batch_norm:
            if self.batch_norm == 'batch':
                out = self.module._modules['batch_norm_'+self.name_suffix](out)
            elif self.batch_norm == 'instance':
                out = self.module._modules['instance_norm_'+self.name_suffix](out.unsqueeze(1))
                out = out.squeeze()
        if self.nn_lin:
            out = self.module._modules['nnlin'+self.name_suffix](out)
        return out


class MLPResidualLayer(MLPLayer):
    def forward(self, x):
        out = super(MLPResidualLayer, self).forward(x)
        if self.input_dim == self.output_dim:
            out = nn.functional.relu(out + x)
        return out


class MLP(nn.Module):
    ''' Generic layer that is used by generative variational models as encoders, decoders or only hidden layers.'''
    def __init__(self, pins, phidden, pouts=None, name="", device='cpu', make_flows=True, *args, **kwargs):
        ''':param pins: Input properties.
        :type pins: dict or [dict]
        :param pouts: Out propoerties. Leave to None if you only want hidden modules.
        :type pouts: [dict] or None
        :param phidden: properties of hidden layers.
        :type phidden: dict
        :param nn_lin: name of non-linear layer 
        :type nn_lin: string
        :param name: name of module
        :type name: string'''
        
        super(MLP, self).__init__()
        self.device = device
        # set input parameters
        if not issubclass(type(pins), list):
            pins = [pins]
        self.pins = pins
        self.phidden = phidden
        
        # get hidden layers
        self.hidden_module = self.get_hidden_layers(pins, phidden, name)
        
            
    def get_hidden_layers(self, pins, phidden={"dim":800, "nlayers":2, 'label':None, 'nn_lin':'ReLU', 'conditioning':'concat'}, name=""):
        ''' outputs the hidden module of the layer.
        :param input_dim: dimension of the input
        :type input_dim: int
        :param phidden: parameters of hidden layers
        :type phidden: dict
        :param nn_lin: non-linearity name
        :type nn_lin: str
        :param name: name of module
        :type name: str
        :returns: nn.Sequential '''
        # Hidden layers
            
        residual = phidden.get("residual", False)
        nn_lin = phidden.get('nn_lin', 'ReLU')
        LayerModule = MLPResidualLayer if residual else MLPLayer
        
        input_dim = 0
        for p in pins:
            input_dim += p['dim'] if issubclass(type(p), dict) else p
            
        nlayers = phidden.get('nlayers', 1)
        hidden_dims = phidden.get('dim', [800]*nlayers)
        if not issubclass(type(hidden_dims), list):
            hidden_dims = [hidden_dims]*nlayers
        if not issubclass(type(phidden), list):     
            # add dimensions in  case of concatenative conditioning
            is_conditioned = phidden.get('label')
            conditioning = phidden.get('conditioning', 'concat')
            
#            pdb.set_trace()
            if is_conditioned and conditioning == 'concat':
                label_dim = sum([x['dim'] for  x in is_conditioned]) if issubclass(type(is_conditioned), list) else is_conditioned['dim']
                input_dim += label_dim
            
            modules = OrderedDict()
            for i in range(nlayers):
                hidden_dim = hidden_dims[i]
                n_in = int(input_dim) if i==0 else int(hidden_dims[i-1])
                modules['layer_%d'%i] = LayerModule(n_in, int(hidden_dim), nn_lin=nn_lin, batch_norm=phidden.get('batch_norm', False), dropout = phidden.get('dropout'), name_suffix="_%d"%i)
            hidden_module = nn.Sequential(modules)
        else:
            new_modules = [self.get_hidden_layers(self, pins, ph) for ph in self.phidden]
            hidden_module = nn.ModuleList(new_modules)
            
        return hidden_module
                
        
    def format_label_data(self, y, phidden):
        def process(y, plabel):
            if type(y)==np.ndarray:
                if y.ndim == 1:
                    y = utils.oneHot(y, plabel['dim'])
                y = torch.from_numpy(y)
            device = next(self.parameters()).device     
            y = y.to(device, dtype=torch.float32)
            return y
        
        if not self.phidden.get('label') or y is None:
            return
        
        if issubclass(type(self.phidden['label']), list):
            y = tuple([process(y[plabel['task']].copy(), plabel) for plabel in phidden['label']])
            y = torch.cat(y, 1)    
        else:
            y = process(y[self.phidden['label']['task']].copy(), phidden['label'])  
            
        return y

                
    def forward(self, x, y=None, sample=True, *args, **kwargs):
        '''outputs parameters of corresponding output distributions
        :param x: input or vector of inputs.
        :type x: torch.Tensor or [torch.Tensor ... torch.Tensor]
        :param outputHidden: also outputs hidden vector
        :type outputHidden: True
        :returns: (torch.Tensor..torch.Tensor)[, torch.Tensor]'''
                
        # Concatenate input if module is a list
        if type(x)==list:
            ins = cat(x, 1)
        else:
            ins = x
                     
        # Distributes input in case of multiple hidden modules
        if issubclass(type(self.hidden_module), nn.ModuleList):
            h = []
            for i, hm in enumerate(self.phidden):
                ins_tmp = ins
                if hm.get('label') and hm.get('conditioning', 'concat'):
                    assert y, 'need label if hidden module is conditioned'
                    y_tmp = self.format_label_data(y, hm)
                    ins_tmp = cat((x, y_tmp), 1)
                    
                h.append(hm[i](ins_tmp))
        else:
            if self.phidden.get('label') and self.phidden.get('conditioning', 'concat'):
                assert y, 'need label if hidden module is conditioned'
                y_tmp = self.format_label_data(y, self.phidden)
                ins = cat((x, y_tmp), 1)
            h = self.hidden_module(ins)
            
        return h

    
    
    
    
    
    
class DLGMLayer(nn.Module):
    ''' Specific decoding module for Deep Latent Gaussian Models'''
    def __init__(self, pins, pouts, nn_lin="ReLU", name="", **kwargs):
        '''
        :param pins: parameters of the above layer
        :type pins: dict
        :param pouts: parameters of the ouput distribution
        :type pouts: dict
        :param phidden: parameters of the hidden layer(s)
        :type phidden: dict
        :param nn_lin: non-linearity name
        :type nn_lin: str
        :param name: name of module
        :type name: str'''
        
        super(DLGMLayer, self).__init__()
                
        if issubclass(type(pouts), list):
            self.out_module = nn.ModuleList()
            self.cov_module = nn.ModuleList()
            for pout in pouts:
                self.out_module.append(nn.Linear(pins['dim'], pout['dim']))
                init_module(self.out_module, 'Linear')
                self.cov_module.append(nn.Sequential(nn.Linear(pout['dim'], pout['dim']), nn.Sigmoid()))
                init_module(self.cov_module, 'Sigmoid')
        else:
            self.out_module = nn.Linear(pins['dim'], pouts['dim'])
            init_module(self.out_module, 'Linear')
            self.cov_module = nn.Sequential(nn.Linear(pouts['dim'], pouts['dim']), nn.Sigmoid())
            init_module(self.cov_module, 'Sigmoid')
        
        
        
    def forward(self, h, eps, sample=True):
        '''outputs the latent vector of the corresponding layer
        :param z: latent vector of the above layer
        :type z: torch.Tensor
        :param eps: latent stochastic variables
        :type z: torch.Tensor
        :returns:torch.Tensor'''
            
        if issubclass(type(h), list):
            h = cat(tuple(h), 1)
        
        if issubclass(type(self.out_module), nn.ModuleList):
            params = []
            for i, module in enumerate(self.out_module):
                params.append(self.out_module[i](h), self.cov_module[i](eps))
        else:
            params = (self.out_module(h), self.cov_module(eps))
            
        return params
        
    
