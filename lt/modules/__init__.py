import torch.nn as nn

MLP_DEFAULT_NNLIN = "ReLU"
CONV_DEFAULT_NNLIN = "ReLU"
DEFAULT_INIT = nn.init.xavier_normal_

def get_init(nn_lin):
    if nn_lin=="ReLU":
        return 'relu'
    elif nn_lin=="TanH":
        return 'tanh'
    elif nn_lin=="LeakyReLU":
        return 'leaky_relu'
    elif nn_lin=="conv1d":
        return "conv1d"
    elif nn_lin=="cov2d":
        return "conv2d"
    elif nn_lin=="conv3d":
        return "conv3d"
    elif nn_lin=="Sigmoid":
        return "sigmoid"
    else:
        return "linear"
    
def init_module(module, nn_lin=MLP_DEFAULT_NNLIN, method=DEFAULT_INIT):
    gain = nn.init.calculate_gain(get_init(nn_lin))
    if type(module)==nn.Sequential:
        for m in module:
            init_module(m, nn_lin=nn_lin, method=method)
    if type(module)==nn.Linear:
        method(module.weight.data, gain)
        nn.init.zeros_(module.bias)

class Identity(nn.Module):
    def __call__(self, *args, **kwargs):
        return args


from . import modules_bottleneck as bottleneck
from . import modules_convolution as convolution
from . import modules_distribution as distributions
from . import modules_vanillaVAE as vanillaVAE
#from . import modules_flows as flows

from . import modules_hidden as hidden
from .. import utils as utils


