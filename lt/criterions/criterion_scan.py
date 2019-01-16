import torch.nn.functional as func
from . import ELBO

class SCANLoss(ELBO):
    def __init__(self, options = {'distance':'kld'}):
        super(SCANLoss, self).__init__(options)
        self.distance_type = options.get('distance', 'kld')
        self.beta = options.get('beta', [1.0, 1.0])
        self.cross_factors = options.get('cross_factors', [10.0, 0.0])

    def get_transfer_error(self, models, outs, distance_type):
        if distance_type=='kld':
            tferr1 = self.get_kld(outs[0]['z_params_enc'][-1], outs[1]['z_params_enc'][-1], models[0].platent[-1]['dist'], models[1].platent[-1]['dist'])
            tferr2 = self.get_kld(outs[1]['z_params_enc'][-1], outs[0]['z_params_enc'][-1], models[1].platent[-1]['dist'], models[0].platent[-1]['dist'])
        elif distance_type=='l2':
            tferr1 = func.mse_loss(outs[0]['z_params_enc'][-1][0], outs[1]['z_params_enc'][-1][0], size_average = self.size_average)
            tferr2 = tferr1.clone()
            if not self.size_average:
                tferr1 /= outs[0]['z_params_enc'][-1][0].shape[0]
                tferr2 /= outs[0]['z_params_enc'][-1][0].shape[0]
        return tferr1, tferr2

    def loss(self, models, outs, xs=None, epoch=None, options = {}, write=None, *args, **kwargs):
        assert not xs is None  
        beta = options.get('betas', self.beta)
        distance_type = options.get('distance', self.distance_type)
        cross_factors = options.get('betas', self.cross_factors)
#        if not epoch is None and self.warmup != 0:
#            beta_1 = beta_1 * min(epoch / self.warmup, 1.0)
            
        # reconstruction errors
        rec_errors = [self.get_reconstruction_error(models[i], xs[i], outs[i], *args, **kwargs) for i in range(len(models))]
        # kld errors
        kld_errors = [self.get_regularization_error(models[i],  outs[i], *args, **kwargs)[0] for i in range(len(models))]
        # transfer errors
        tferr1, tferr2 = self.get_transfer_error(models, outs, distance_type)
        tferr1 = self.get_kld(outs[0]['z_params_enc'][-1], outs[1]['z_params_enc'][-1], models[0].platent[-1]['dist'], models[1].platent[-1]['dist'])
        tferr2 = self.get_kld(outs[1]['z_params_enc'][-1], outs[0]['z_params_enc'][-1], models[1].platent[-1]['dist'], models[0].platent[-1]['dist'])
        
        loss = sum(rec_errors) + sum([beta[i] * kld_errors[i] for i in range(len(kld_errors))]) + cross_factors[0]*tferr1 + cross_factors[1]*tferr2
        losses = (rec_errors[0],rec_errors[1], kld_errors[0], kld_errors[1], tferr1, tferr2)
        
        if write:
            self.write(write, losses)
            
        return loss, losses


    def get_named_losses(self, losses):
        dict_losses = {'rec_loss_1':losses[0].item(), 'kld_prior_1':losses[1].item(),
                       'rec_loss_2':losses[2].item(), 'kld_prior_2':losses[3].item(),
                       'transfer_kld_1':losses[0].item(), 'transfer_kld_2':losses[1].item()}
        return dict_losses

