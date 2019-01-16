import torch
from torch.nn import functional as F
from torch.autograd import Variable

from numpy import pi, log
import torch.distributions as dist

from ..utils import onehot


# log-probabilities
def log_bernoulli(x, x_params, size_average=False):
    if torch.__version__ == '0.4.1':
        if size_average:
            size_average = 'elementwise_mean'
        else:
            size_average = 'sum'
        loss = F.binary_cross_entropy(x_params[0], x, reduction=size_average)
    else:
        loss = F.binary_cross_entropy(x_params[0], x, size_average=size_average)

    if not size_average:
        loss = loss / x.size(0)
    return loss
    #return F.binary_cross_entropy(x_params[0], x, size_average = False)

def log_normal(x, x_params, logvar=False, clamp=True, size_average=False):
    x = x.squeeze()
    if x_params == []:
        x_params = [torch.zeros_like(0, device=x.device), torch.zeros_like(0, device=x.device)]
    if len(x_params)<2:
        x_params.append(torch.full_like(x_params[0], 1e-3, device=x.device))
    mean, std = x_params
    if not logvar:
        std = std.log()
    # average probablities on batches
    #result = torch.mean(torch.sum(0.5*(std + (x-mean).pow(2).div(std.exp())+log(2*pi)), 1))
    loss = 0.5*(std + (x-mean).pow(2).div(std.exp())+log(2*pi))

    if size_average:
        loss = torch.mean(loss)
    else:
        loss = torch.mean(torch.sum(loss, 1))
    #result = F.mse_loss(x, x_params[0])
    return loss

def log_categorical(y, y_params, size_average=False):
    loss = F.nll_loss(y_params[0], onehot.fromOneHot(y).long(), size_average=size_average)
    if not size_average:
        loss = loss / y.size(0)
    return loss

def log_density(in_dist):
    if in_dist in [dist.Bernoulli]:
        return log_bernoulli
#    elif in_dist.dist_class==dist.normal.dist_class or in_dist.dist_class==cust.spectral.dist_class:
    elif in_dist in [dist.Normal]:
        return log_normal
    elif in_dist==dist.Categorical:
        return log_categorical
    else:
        raise Exception("Cannot find a criterion for distribution %s"%in_dist)
