import torch
import torch.nn as nn
import collections 

class AdversarialLayer(nn.Module):
    def __init__(self, input_params, hidden_params):
        super(AdversarialLayer, self).__init__()
        self.pinput = input_params
        self.phidden = hidden_params
        self.hidden_module = self.get_hidden_module(input_params, hidden_params)
        
    def get_hidden_module(self, input_params, hidden_params):
        modules = collections.OrderedDict()
        for i in range(hidden_params['nlayers']):
            if i == 0:
                modules['layer_%d'%i] = nn.Linear(input_params['dim'], hidden_params['dim'])
            else:
                modules['layer_%d'%i] = nn.Linear(hidden_params['dim'], hidden_params['dim'])
            modules['batch_norm'] = nn.BatchNorm1d(hidden_params['dim'])
            modules['nn_lin_%d'%i] = nn.ReLU()
        modules['final_layer'] = nn.Linear(hidden_params['dim'], 1)
        modules['nn_lin'] = nn.Sigmoid()
        return nn.Sequential(modules)
    
    def forward(self, input):
        return self.hidden_module(input)
    
    def init_optimizer(self, optim_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optim_args)
        
    def update_optimizers(self, args):
        self.optimizer.zero_grad()
        
    def step(self, loss):
        self.optimizer.step()
        
    