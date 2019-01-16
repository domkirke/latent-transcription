# -*- coding: utf-8 -*-

import torch.distributions as dist
from .prior_prior import Prior
from torch import zeros, ones

class IsotropicGaussian(Prior):
    def __init__(self, dim, *args, **kwargs):
        super(IsotropicGaussian, self).__init__()
        self.params = (zeros((1, dim)),
                       ones((1, dim)))
        self.params[0].requires_grad_(False)
        self.params[1].requires_grad_(False)
        self.dist = dist.Normal
        
    def __call__(self, n_batches, requires_grad = True):
        eps = self.dist(self.params[0].repeat(n_batches, 1), self.params[1].repeat(n_batches, 1)).rsample()
        eps.requires_grad_(requires_grad)
        print(eps.requires_grad)
        return eps


class DiagonalGaussian(Prior):
    def __init__(self, params):
        assert len(params)==2
        self.dim = params[0].size(1)
        self.dist = dist.Normal
        self.params = params
        
