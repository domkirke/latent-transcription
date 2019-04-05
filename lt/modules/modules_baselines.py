import torch
import torch.nn as nn
from .modules_bottleneck import MLP
from .modules_convolution import Convolutional, DeconvolutionalLatent


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


class Signal2SymbolConvNet(nn.Module):
    def __init__(self, input_params, hidden_params, output_params):
        super(Signal2SymbolConvNet, self).__init__()
        self.pinput = input_params
        self.phidden = hidden_params
        self.poutput = output_params
        self.conv_module = Convolutional(input_params, hidden_params)
        conv_out_shape = self.phidden['channels'][-1] * self.conv_module.get_output_conv_length()[0][-1]
        out_modules = []
        for out in output_params:
            out_modules.append(nn.Sequential(MLP({'dim':conv_out_shape}, self.phidden), nn.Linear(self.phidden['dim'], out['dim']), nn.LogSoftmax()))
        self.out_modules = nn.ModuleList(out_modules)

    def forward(self, x):
        conv_out = self.conv_module(x)
        conv_out = conv_out.view(conv_out.shape[0], conv_out.shape[1]*conv_out.shape[2])
        outs = [out_module(conv_out) for out_module in self.out_modules]

        return outs


class Symbol2SignalConvNet(nn.Module):
    def __init__(self, input_params, hidden_params, output_params):
        super(Symbol2SignalConvNet, self).__init__()
        input_params = {'dim':sum([i['dim'] for i in input_params])}
        self.pinput = input_params
        self.phidden = hidden_params
        self.poutput = output_params
        self.conv_module = DeconvolutionalLatent(input_params, hidden_params, pouts=output_params)

    def forward(self, x):
        conv_out = self.conv_module(x.squeeze())
        return conv_out

