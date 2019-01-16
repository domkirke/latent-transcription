#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:38:11 2017

@author: chemla
"""

import pdb
import torch.nn as nn
import torch.optim

from ..modules.modules_hidden import  HiddenModule
from .modules_abstractVAE import AbstractVAE 


class VanillaVAE(AbstractVAE):
    # initialisation of the VAE
    def __init__(self, input_params, latent_params, hidden_params = {"dim":800, "nlayers":2}, *args, **kwargs):
        super(VanillaVAE, self).__init__(input_params, latent_params, hidden_params, *args, **kwargs)    
        
    def make_encoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        encoders = nn.ModuleList()
        for layer in range(len(latent_params)):
            if layer==0:
                encoders.append(self.make_encoder(input_params, latent_params[0], hidden_params[0], name="vae_encoder_%d"%layer))
            else:
                encoders.append(self.make_encoder(latent_params[layer-1], latent_params[layer], hidden_params[layer], nn_lin="ReLU", name="vae_encoder_%d"%layer))
        return encoders
    
    
    @classmethod
    def make_encoder(cls, input_params, latent_params, hidden_params, *args, **kwargs):               
        kwargs['name'] = kwargs.get('name', 'vae_encoder')
#        ModuleClass = hidden_params.get('class', DEFAULT_MODULE)
#        module = latent_params.get('shared_encoder') or ModuleClass(input_params, latent_params, hidden_params, *args, **kwargs)
        module = HiddenModule(input_params, hidden_params, latent_params, *args, **kwargs)
        return module
    
    def make_decoders(self, input_params, latent_params, hidden_params, extra_inputs=[], *args, **kwargs):
        decoders = nn.ModuleList()
        for layer in range(len(latent_params)):
            if layer==0:
                #TODO pas terrible d'embarquer l'encoder comme ça
                new_decoder = self.make_decoder(input_params, latent_params[0], hidden_params[0], name="vae_decoder_%d"%layer, encoder = self.encoders[layer])
            else:
                new_decoder = self.make_decoder(latent_params[layer-1], latent_params[layer], hidden_params[layer], name="vae_decoder_%d"%layer, encoder=self.encoders[layer])
            decoders.append(new_decoder)
        return decoders
    
    @classmethod
    def make_decoder(cls, input_params, latent_params, hidden_params, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'vae_decoder')
#        ModuleClass = hidden_params.get('class', DEFAULT_MODULE)
#        module = hidden_params.get('shared_decoder') or ModuleClass(latent_params, input_params, hidden_params, *args, **kwargs)
        module = HiddenModule(latent_params, hidden_params, input_params, make_flows=False, *args, **kwargs)
        return module
        
    
    # processing methods
    def encode(self, x, y=None, sample=True, from_layer=0, *args, **kwargs):
        ins = x; outs = []
        for layer in range(from_layer, len(self.platent)):
            module_out = self.encoders[layer](ins, y=y, *args, **kwargs)
            outs.append(module_out)
            if sample:
                ins = outs[-1]['out']
            else:
                ins = outs[-1]['out_params'][0]
        return outs
        
    def decode(self, z, y=None, sample=True, from_layer=-1, *args, **kwargs):
        if from_layer < 0:
            from_layer += len(self.platent)
        ins = z; outs = []
        for i,l in enumerate(reversed(range(from_layer+1))):
            module_out = self.decoders[l](ins, y=y)
            outs.append(module_out)
            if sample:
                ins = outs[-1]['out']
            else:
                ins = outs[-1]['out_params'][0]
        outs = list(reversed(outs))
        return outs
            
    
    def forward(self, x, y=None, options={}, *args, **kwargs):
        def denest_dict(nest_dict):
            keys = set()
            new_dict = {}
            for item in nest_dict:
                keys = keys.union(set(item.keys()))
            for k in keys:    
                new_dict[k] = [x[k] for x in nest_dict]
            return new_dict
        
        x = self.format_input_data(x)
        enc_out = self.encode(x, y=y, *args, **kwargs)
        dec_out = self.decode(enc_out[-1]['out'], y=y, *args, **kwargs)
        
        x_params = dec_out[0]['out_params']
        dec_out = denest_dict(dec_out[1:]) if len(dec_out) > 1 else {}
        enc_out = denest_dict(enc_out)       
        return {'x_params':x_params, 
                'z_params_dec':dec_out.get('out_params'), 'z_dec':dec_out.get('out'),
                'z_params_enc':enc_out['out_params'], 'z_enc':enc_out['out']}
    
    
    # define optimizer
    def init_optimizer(self, optim_params):
        self.optim_params = optim_params
        alg = optim_params.get('optimizer', 'Adam')
        optim_args = optim_params.get('optim_args', {'lr':1e-5})
        optimization_mode = optim_params.get('optimize', 'full');
        if optimization_mode == 'full':
            self.optimizers = {'default':getattr(torch.optim, alg)(self.parameters(), **optim_args)}   
        elif optimization_mode == 'enc':
            self.optimizers = {'default':getattr(torch.optim, alg)(self.encoders.parameters(), **optim_args)}   

        
        scheduler = optim_params.get('scheduler', 'ReduceLROnPlateau')
        scheduler_args = optim_params.get('scheduler_args', {'patience':100, "factor":0.2, 'eps':1e-10})
        self.schedulers = {'default':getattr(torch.optim.lr_scheduler, scheduler)(self.optimizers['default'], **scheduler_args)} 
        
        
    def optimize(self, loss, options={}, retain_graph=False, *args, **kwargs):
        # optimize
        loss.backward(retain_graph=retain_graph)
        self.optimizers['default'].step()
        
    # define losses 
    

    

        

            
