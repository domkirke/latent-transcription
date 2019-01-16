#from pyro.nn import AutoRegressiveNN
from numpy import ceil, array
import torch.nn as nn, pdb
from .modules_utils import init_module
import torch.distributions as dist
import sys
sys.path.append('..')

#from . import modules_convolution as conv


########################################################################
####        Gaussian layers

class GaussianLayer(nn.Module):
    def __init__(self, pinput, poutput, **kwargs):
        if issubclass(type(poutput['dim']), list):
            if len(poutput['dim']) == 2:
                GaussianLayer2D.__init__(self, pinput, poutput, **kwargs)
                self.forward_method = GaussianLayer2D.forward
                self.requires_deconv_indices = GaussianLayer2D.requires_deconv_indices
        else:
            print(poutput)
            if poutput.get('conv'):
                GaussianLayer2D.__init__(self, pinput, poutput, **kwargs)
                self.forward_method = GaussianLayer2D.forward
                self.requires_deconv_indices = GaussianLayer2D.requires_deconv_indices
            else:
                GaussianLayer1D.__init__(self, pinput, poutput, **kwargs)
                self.forward_method = GaussianLayer1D.forward
                self.requires_deconv_indices = GaussianLayer1D.requires_deconv_indices
            
    def forward(self, *args, **kwargs):
        return self.forward_method(self, *args, **kwargs)
                
            
class GaussianLayer1D(nn.Module):
    requires_deconv_indices = False
    '''Module that outputs parameters of a Gaussian distribution.'''
    def __init__(self, pinput, poutput, **kwargs):
        '''Args
            pinput (dict): dimension of input
            poutput (dict): dimension of output
        '''
        nn.Module.__init__(self)
        self.input_dim = pinput['dim']; self.output_dim = poutput['dim']
        self.modules_list = nn.ModuleList()
        self.nn_lin = poutput.get('nn_lin')
        if issubclass(type(self.input_dim), list):
            input_dim_mean = self.input_dim[0]
            input_dim_var = self.input_dim[1]
        else:
            input_dim_mean = input_dim_var = self.input_dim
        mean_module = nn.Linear(input_dim_mean, self.output_dim)
        init_module(mean_module, 'Linear')
        self.modules_list.append(mean_module)
        var_module = nn.Sequential(nn.Linear(input_dim_var, self.output_dim), nn.Sigmoid())
        init_module(var_module, 'Sigmoid')
        self.modules_list.append(var_module)
        
    def forward(self, ins,  *args, **kwargs):
        '''Outputs parameters of a diagonal Gaussian distribution.
        :param ins : input vector.
        :returns: (torch.Tensor, torch.Tensor)'''
        if self.nn_lin:
            nn_lin = getattr(nn.functional, self.nn_lin)
            mu = nn_lin(self.modules_list[0](ins))
        else:
            mu = self.modules_list[0](ins)
        logvar = self.modules_list[1](ins)
        return mu, logvar
    
           
        
        
        
########################################################################
####        Bernoulli layers
      
        
class BernoulliLayer(nn.Module):
    def __init__(self, pinput, poutput, **kwargs):
        if issubclass(type(poutput['dim']), list):
            if len(poutput) == 2:
                BernoulliLayer2D.__init__(self, pinput, poutput, **kwargs)
                self.forward_method = BernoulliLayer2D.forward
                self.requires_deconv_indices = BernoulliLayer2D.requires_deconv_indices
        else:
            BernoulliLayer1D.__init__(self, pinput, poutput, **kwargs)
            self.forward_method = BernoulliLayer1D.forward
            self.requires_deconv_indices = BernoulliLayer1D.requires_deconv_indices
            
    def forward(self, *args, **kwargs):
        return self.forward_method(self, *args, **kwargs)
                


class BernoulliLayer1D(nn.Module):
    requires_deconv_indices = False
    '''Module that outputs parameters of a Bernoulli distribution.'''
    def __init__(self, pinput, poutput, **kwargs):
        super(BernoulliLayer, self).__init__()
        self.input_dim = pinput['dim']; self.output_dim = poutput['dim']
        self.modules_list = nn.Sequential(nn.Linear(self.input_dim, self.output_dim), nn.Sigmoid())
        init_module(self.modules_list, 'Sigmoid')
        
    def forward(self, ins,  *args, **kwargs):
        mu = self.modules_list(ins)
        return (mu,) 
   
########################################################################
####        Categorical layers


class CategoricalLayer(nn.Module):
    def __init__(self, pinput, poutput, **kwargs):
        if issubclass(type(poutput['dim']), list):
            if len(poutput) == 2:
                CategoricalLayer2D.__init__(self, pinput, poutput, **kwargs)
                self.forward_method = CategoricalLayer2D.forward
                self.requires_deconv_indices = CategoricalLayer2D.requires_deconv_indices
        else:
            CategoricalLayer1D.__init__(self, pinput, poutput, **kwargs)
            self.forward_method = CategoricalLayer1D.forward
            self.requires_deconv_indices = CategoricalLayer1D.requires_deconv_indices
            
    def forward(self, *args, **kwargs):
        return self.forward_method(self, *args, **kwargs)
                


class CategoricalLayer1D(nn.Module):
    requires_deconv_indices = False
    '''Module that outputs parameters of a Bernoulli distribution.'''
    def __init__(self, pinput, poutput, **kwargs):
        nn.Module.__init__(self)
        self.input_dim = pinput['dim']; self.output_dim = poutput['dim']
        self.modules_list = nn.Sequential(nn.Linear(self.input_dim, self.output_dim), nn.Softmax())
        init_module(self.modules_list, 'Sigmoid')
        
    def forward(self, ins,  *args, **kwargs):
        mu = self.modules_list(ins)
        return (mu,) 
 
def get_module_from_density(distrib):
    if distrib == dist.Normal:
        return GaussianLayer
    elif distrib == dist.Bernoulli:
        return BernoulliLayer
    elif distrib == dist.Categorical:
        return CategoricalLayer
    elif distrib == cust.Spectral:
        return SpectralLayer
    elif distrib == cust.AutoRegressive(dist.normal):
        return AutoRegressiveGaussianLayer
    elif distrib == cust.AutoRegressive(dist.bernoulli):
        return AutoRegressiveBernoulliLayer
    else:
        raise TypeError('Unknown distribution type : %s'%distrib)
