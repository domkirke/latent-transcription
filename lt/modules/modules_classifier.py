import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, outputs, hidden_dims=5000, hidden_layers=2):
        super(Classifier, self).__init__()
        self.hidden_module = self.get_hidden_module(input_dim, hidden_dims=hidden_dims, hidden_layers=hidden_layers)
        self.output_modules = self.get_output_modules(hidden_dims, outputs)
        
    def get_hidden_module(self, input_dim, hidden_dims=5000, hidden_layers=2):
        module_list = []
        for i in range(hidden_layers):
            i1 = input_dim if i == 0 else hidden_dims
            i2 = hidden_dims
            module_list.append(nn.Linear(i1,i2))
            module_list.append(nn.BatchNorm1d(i2))
            module_list.append(nn.ReLU())
        module_list = tuple(module_list)
        return nn.Sequential(*module_list)
    
    def get_output_modules(self, hidden_dims, outputs):
        output_modules = nn.ModuleList()
        for o in outputs:
            if o['dist'] == torch.distributions.Bernoulli:
                module = nn.Sequential(nn.Linear(hidden_dims, o['dim']), nn.Sigmoid())
                output_modules.append(module)
            elif o['dist'] == torch.distributions.Categorical:
                module = nn.Sequential(nn.Linear(hidden_dims, o['dim']), nn.Softmax())
                output_modules.append(module)
        return output_modules
    
    def forward(self, x):
        hidden_layer = self.hidden_module(x)
        outs = []
        for output_module in self.output_modules:
            outs.append(output_module(hidden_layer))
        return outs
