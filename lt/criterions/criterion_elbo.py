import torch, pdb
import numpy as np
from . import log_density, kld
from ..distributions.priors import IsotropicGaussian
from . import Criterion


def checklist(item):
    if not issubclass(type(item), list):
        item = [item]
    return item



class MSE(Criterion):
    def __init__(self, options={}):
        super(MSE, self).__init__(options)
        self.size_average = options.get('size_average', False)
        
    def loss(self, out, x=None, write=False):
        assert not x is None
        if issubclass(type(out), list):
            losses = [self.loss(out[i], x[i]) for i in range(len(out))]
            loss = sum(losses)        
        else:
            loss = torch.nn.functional.mse_loss(out, x, size_average=self.size_average)
        if write:
            self.write(write, (loss,))

        return loss, (loss,)
    
    def get_named_losses(self, losses):
        return {'mse_loss':losses[0].item()}



class ELBO(Criterion):
    def __init__(self, options={}):
        super(ELBO, self).__init__(options)
        self.warmup = options.get('warmup', 100)
        self.beta = options.get('beta', 1.0)
        self.size_average = options.get('size_average', False)
        self.epoch = -1
        
        
    def get_reconstruction_error(self, model, x, out, *args, **kwargs):
        # reconstruction error
        if issubclass(type(model.pinput), list):
            rec_error = torch.zeros(1, requires_grad=True, device=model.device)
            pinput = model.pinput
            if not issubclass(type(x), list):
                x = [x]
            for i in range(len(x)):
                rec_error = rec_error + log_density(pinput[i]['dist'])(x[i], out['x_params'][i], size_average=self.size_average)
        else:
            rec_error = log_density(model.pinput['dist'])(x, out['x_params'], size_average=self.size_average)
        return rec_error
    
    
    def get_kld(self, p_enc, p_dec, enc_dist, dec_dist, *args, **kwargs):
        kld_error = kld(enc_dist, dec_dist)(p_enc, p_dec, size_average=self.size_average)
        return kld_error


    def get_montecarlo(self, model, out, *args, **kwargs):
        kld_error = torch.zeros(1, requires_grad=True, device=model.device)
        # turn parameters into lists
        current_zenc =  checklist(out['z_enc'][l])                  # sampled z
        current_zenc_params = checklist(out['z_params_enc'][l])     # z params
        platent_tmp = checklist(platent_tmp)                        # current layer's parameters
        # enumerate over layers
        for i, pl in enumerate(platent_tmp):
            # enumerate over splitted latent variables 
            # get p(z | prior)
            if l == len(model.platent)-1:
                prior = pl.get('prior') or IsotropicGaussian(pl['dim'])
                log_p = log_density(prior.dist)(current_zenc[i], prior.get_params(device=model.device, *args, **kwargs))
            else:
                current_zdec_params = checklist(out['z_params_dec'][l])
                log_p = log_density(pl['dist'])(current_zenc[i], current_zdec_params[i], size_average=self.size_average)
            # get q(z | z_params)
            log_q = log_density(pl['dist'])(current_zenc[i],  current_zenc_params[l], size_average=self.size_average)
            # compute kld component
            kld_error = kld_error + log_p - log_q
        return kld_error

    
    def get_regularization_error(self, model, out, *args, **kwargs):
        sample = kwargs.get('sample', False)
        kld_errors = []
        platent = model.platent
        if not issubclass(type(model.platent), list):
            platent = [platent]
        # iterate over layers
        for l, platent_tmp in enumerate(platent):
            # in case of splitted latent variables
            if not issubclass(type(platent_tmp), list):
                platent_tmp = [platent_tmp]
            err = 0
            for subl, pl in enumerate(platent_tmp):
                sample_tmp = sample and (not platent_tmp.get('flow'))
                if sample_tmp:
                    err += self.get_montecarlo(out['z_enc'][l], platent_tmp, *args, **kwargs)
                else:
                    # get encoder parameters
                    enc_params = out['z_params_enc'][l]
                    enc_dist = pl['dist']
                    if issubclass(type(enc_params), list):
                        enc_params = enc_params[subl]
#                    if l == len(model.platent)-1:
#                        prior = pl.get('prior') or IsotropicGaussian(pl['dim'])
#                        dec_params = prior.get_params(device = model.device, *args, **kwargs)
#                        dec_dist = prior.dist
#                    else:
#                        pdb.set_trace()
#                        dec_params = out['z_params_dec'][l]
#                        dec_dist = pl['dist']
                    prior = pl.get('prior') or IsotropicGaussian(pl['dim'])
                    dec_params = prior.get_params(device = model.device, *args, **kwargs)
                    dec_dist = prior.dist
                    err += self.get_kld(enc_params, dec_params, enc_dist, dec_dist, *args, **kwargs)
                if pl.get('flows'):
                    err -= self.get_flow_error(model.encoders[l].flows[subl])
            kld_errors.append(err)
            
        return kld_errors
    
    
    def get_flow_error(self, module):
        logdets = []
        for flow in module : 
            logdets.append(torch.mean(flow.current_det))
        return sum(logdets)
        
            
    def loss(self, model, out, x = None, epoch = None, options = {}, write=None, rec_loss=True, kld_loss=True, *args, **kwargs):
        assert not x is None        
        beta = options.get('beta', 1.0)
        if not epoch is None and self.warmup != 0:
            beta = beta * min(epoch / self.warmup, 1.0)
        if rec_loss:
            rec_error = self.get_reconstruction_error(model, x, out, *args, **kwargs)
        else:
            rec_error = torch.tensor(0.)
        if kld_loss:
            kld_errors = self.get_regularization_error(model, out, *args, **kwargs)
            kld_error = sum(kld_errors)
        else:
            kld_error = torch.tensor(0.)
        loss = rec_error + beta * kld_error     
        losses = (rec_error, kld_error)
        if write:
            self.write(write, losses)
        return loss, losses
        
            
    def get_named_losses(self, losses):
        dict_losses = {'rec_loss':losses[0].item(), 'kld_loss':losses[1].item()}
        return dict_losses
    
    def get_min_loss(self, loss_names=None):
        if loss_names is None:
            loss_names = self.loss_history.keys()
        added_losses = []
        for loss_name in loss_names:
            added_losses.append(self.loss_history[loss_name])
        losses = np.sum(np.array(added_losses), 1)
        return torch.min(losses)
            
    
        
        
