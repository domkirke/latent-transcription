#!/usr/bin/env python3
"""
# -*- coding: utf-8 -*-
Created on Mon Oct 22 14:26:52 2018

@author: chemla
"""

# -*- coding: utf-8 -*-
import argparse, copy, os, gc, sys

import matplotlib
matplotlib.use('agg')

import numpy as np

import torch
import torch.nn as nn
import torch.distributions as dist
#import matplotlib
#matplotlib.use('agg')

from lt.modules.modules_vanillaVAE import VanillaVAE
from lt.data.audio import DatasetAudio

from lt import data as data
from lt.data.preprocessing import Magnitude
from lt.utils.onehot import oneHot, fromOneHot

import lt.utils.onehot as oh
from lt.utils import get_cmap
from lt.utils.dataloader import MixtureLoader

from lt.criterions.criterion_scan import SCANLoss
from lt.modules.modules_adversarial import AdversarialLayer
import matplotlib.pyplot as plt


#%% Parsing arguments


parser = argparse.ArgumentParser()

# Dataset options
parser.add_argument('--dbroot',         type=str,   default='lt_set.npz',    help='root path of the database (given .npy file)')
parser.add_argument('--savedir', type=str, default='saves', help='output directory')
parser.add_argument('--frames', type=int, nargs="*", default = [10, 25], help="frames taken in each sound file (empty for all file, chunk id or chunk range)")

# Architecture options
parser.add_argument('--dims', type=int, nargs='+', default = [32], help='number of latent dimensions')
parser.add_argument('--hidden_dims_1', type=int, nargs='+', default = [5000], help = 'latent layerwise hidden dimensions for audio vae')
parser.add_argument('--hidden_num_1', type=int, nargs='+', default = [2], help = 'latent layerwise number of hidden layers for audio vae')
parser.add_argument('--hidden_num_2', type=int, nargs='+', default = [2], help = 'latent layerwise number of hidden layers for symbolic vae')
parser.add_argument('--hidden_dims_2', type=int, nargs='+', default = [800], help = 'latent layerwise hidden dimensions for symbolic vae')

# conditioning type
parser.add_argument('--labels', type=str, nargs="*", default=['octave', 'pitch', 'dynamics'], help="name of conditioning labels (octave, pitch, dynamics)")
parser.add_argument('--instruments', type=str, nargs =  '+', default = ['Piano', 'Alto-Sax'], help="name of instruments")
parser.add_argument('--label_type', type=str, choices=['binary', 'categorical'], default='categorical' , help="label conditioning distribution")
parser.add_argument('--regularization_type', type=str, choices=['kld', 'l2'], default='kld', help="latent regularization type between both latent spaces")
parser.add_argument('--random_mode', type=str, choices=['constant', 'bernoulli'], default='constant', help="random weighing of each source in the mixtures")
parser.add_argument('--zero_extra_class', type=int,  default=1, help="has an extra zero class when source is silent (recommanded with bernoulli random_mode)")

# training options
parser.add_argument('--epochs', type=int, default=3000, help="nuber of training epochs")
parser.add_argument('--save_epochs', type=int, default=500, help="saving epochs")
parser.add_argument('--plot_epochs', type=int, default=100, help="plotting epochs")
parser.add_argument('--name', type=str, default='scan_multi', help="name of current training")
parser.add_argument('--cuda', type=int, default=-1, help="cuda id (-1 for cpu)")
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--load_epoch', type=str, default='best')

# loss options 
parser.add_argument('--beta_1', type=float, default = 1.0, help="beta regularization for signal vae")
parser.add_argument('--beta_2', type=float, default = 1.0, help="beta regularization for symbol vae")
parser.add_argument('--cross_1', type=float, default = 10.0, help="cross-regularization between z_signal and z_symbol")
parser.add_argument('--cross_2', type=float, default = 0.0, help="cross-regularization between z_symbol and z_signal")
parser.add_argument('--adversarial', type=int, default=0, help="set to 1 for adversarial reinfocement.")
parser.add_argument('--adv_dim', type=int, default=500, help="hidden capacity for adversarial network")
parser.add_argument('--adv_num', type=int, default=2, help="number of hidden networks for adversarial network")
parser.add_argument('--adv_lr', type=float, default=1e-5, help="learning rate for adversarial network")

args = parser.parse_args()
print(args)



#%% Train function
    
def update_losses(losses_dict, new_losses):
    for k, v in new_losses.items():
        if not k in losses_dict.keys():
            losses_dict[k] = []
        losses_dict[k].append(new_losses[k])
    return losses_dict


def train_model(datasets, meta_datasets, models, loss, adversarial=None, tasks=None, loss_tasks=None, preprocessing=[None,None], options={}, plot_options={}, save_with={}):  
    # Global training parameters
    name = options.get('name', 'model')
    epochs = options.get('epochs', 10000)
    save_epochs = options.get('save_epochs', 100)
    plot_epochs = options.get('plot_epochs', 100)
    batch_size = options.get('batch_size', 64)
    nb_reconstructions = options.get('nb_reconstructions', 3)
    random_mode = options.get('random_mode', 'constant')
    zero_extra_class = options.get('zero_extra_class', 0)
    if loss_tasks is None:
        loss_tasks = tasks if not tasks is None else None

    current_device = next(models[0].parameters()).device
    # Setting results & plotting directories
    results_folder = options.get('results_folder', 'saves/'+name)
    figures_folder = options.get('figures_folder', results_folder+'/figures')
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)
    if not os.path.isdir(results_folder+'/vae_1'):
        os.makedirs(results_folder+'/vae_1')
    if not os.path.isdir(results_folder+'/vae_2'):
        os.makedirs(results_folder+'/vae_2')

        
    # Init training
    epoch = options.get('first_epoch') or 0
    min_test_loss = np.inf; best_model = None
    
    if adversarial:
        adversarial_losses = {'train':[], 'test':[]}
        
    # Start training!
    while epoch < epochs:
        print('-----EPOCH %d'%epoch)
        loader = MixtureLoader(datasets, batch_size=batch_size, partition='train', tasks=tasks, random_mode = random_mode)
        
        # train phase
        batch = 0; current_loss = 0;
        train_losses = None
        models[0].train(); models[1].train()
        for mixture, x, y in loader:
            # format x1
            if not preprocessing[0] is None:
                x1 = preprocessing[0](x)
            x1 = models[0].format_input_data(mixture)
            
            # format x2
            x2 = []
            for d in range(len(meta_datasets)):
                if issubclass(type(meta_datasets[d].data), list):
                    current_metadata = [x_tmp[loader.current_ids[d]] for x_tmp in meta_datasets[d].data] 
                    if zero_extra_class:
                        for i_tmp, tmp in enumerate(current_metadata):
                            current_metadata[i_tmp] = np.concatenate((tmp, np.zeros((tmp.shape[0], 1))), 1)
                            current_metadata[i_tmp][np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata[i_tmp].shape[1]-1)+[1])
                    x2.extend(current_metadata)
                else:
                    current_metadata = meta_datasets[d].data[loader.current_ids[d]]
                    if zero_extra_class:
                        current_metadata = np.concatenate((current_metadata, np.zeros((current_metadata.shape[0], 1))), 1)
                        current_metadata[np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata.shape[1]-1)+[1])
                    x2.append(current_metadata)
              
            x2 = models[1].format_input_data(x2)
                              
            # forward
            outs = [models[0].forward(x1, y=y), models[1].forward(x2, y=y)]
            batch_loss, losses = loss.loss(models, outs, xs=[x1,x2], epoch=epoch, write='train')
            if train_losses is None:
                train_losses = losses 
            else:
                train_losses += losses 
            
            models[0].step(batch_loss, retain_graph=True)
            models[1].step(batch_loss, retain_graph=True)
            
            print("epoch %d / batch %d / losses : %s "%(epoch, batch, loss.get_named_losses(losses)))
            current_loss += batch_loss
            models[0].update_optimizers({}); models[1].update_optimizers({})
            
            # adversarial phase
            if adversarial:
                affectations = torch.distributions.Bernoulli(torch.full((x1.shape[0],1), 0.5)).sample().to(current_device)
                adv_input = torch.where(affectations==1, x1, outs[0]['x_params'][0][0])
                adv_outs = adversarial(adv_input)
                adv_loss = nn.functional.binary_cross_entropy(adv_outs, affectations)
                print("-- adversarial loss : %f"%adv_loss)
                adv_loss.backward()
                adversarial.step(adv_loss)
                adversarial.update_optimizers({})
                adversarial_losses['train'].append(adv_loss.detach().cpu().numpy())
                
                
            batch += 1
            del outs[1]; del outs[0];  
            del x1; del x2; 
            torch.cuda.empty_cache()

            
        current_loss /= batch
        print('--- FINAL LOSS : %s'%current_loss)
        loss.write('train', train_losses)

        if torch.cuda.is_available():
            if current_device != 'cpu':
                if epoch ==0:
                    os.system('echo "epoch %d : allocated %d cache %d \n" > %s/gpulog.txt'%(epoch, torch.cuda.memory_allocated(current_device.index), torch.cuda.memory_cached(current_device.index), results_folder)) 
                else:
                    os.system('echo "epoch %d : allocated %d cache %d \n" >> %s/gpulog.txt'%(epoch, torch.cuda.memory_allocated(current_device.index), torch.cuda.memory_cached(current_device.index), results_folder)) 

        
        ## test_phase
        n_batches = 0
        with torch.no_grad():
            models[0].eval(); models[1].eval()
            loader = MixtureLoader(datasets, batch_size=None, partition='test', tasks=tasks, random_mode = random_mode)
            
            test_loss = None; 
            if adversarial:
                adv_loss = torch.tensor(0., device=next(adversarial.parameters()).device)
            # train phase
            for mixture, x, y in loader:
                # format x1
                n_batches += 1
                if not preprocessing[0] is None:
                    x1 = preprocessing[0](x)
                x1 = models[0].format_input_data(mixture)
                
                # format x2
                x2 = []
                for d in range(len(meta_datasets)):
                    if issubclass(type(meta_datasets[d].data), list):
                        current_metadata = [x_tmp[loader.current_ids[d]] for x_tmp in meta_datasets[d].data] 
                        if zero_extra_class:
                            for i_tmp, tmp in enumerate(current_metadata):
                                current_metadata[i_tmp] = np.concatenate((tmp, np.zeros((tmp.shape[0], 1))), 1)
                                current_metadata[i_tmp][np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata[i_tmp].shape[1]-1)+[1])
                        x2.extend(current_metadata)
                    else:
                        current_metadata = meta_datasets[d].data[loader.current_ids[d]]
                        if zero_extra_class:
                            current_metadata = np.concatenate((current_metadata, np.zeros((current_metadata.shape[0], 1))), 1)
                            current_metadata[np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata[i_tmp].shape[1]-1)+[1])
                        x2.append(current_metadata)
                                  
                # forward
                outs = [models[0].forward(x1, y=y), models[1].forward(x2, y=y)]
                current_test_loss, losses = loss.loss(models, outs, xs=[x1, x2], epoch=epoch, write='train')
                if not test_loss:
                    test_loss = current_test_loss
                else:
                    test_loss = test_loss + current_test_loss
                    
                if adversarial:
                    adv_in = torch.cat((x1, outs[0]['x_params'][0]), 0)
                    adv_out = adversarial(adv_in)
                    adv_target = torch.cat((torch.ones((x1.shape[0], 1), device=current_device), torch.zeros((x1.shape[0], 1), device=current_device)), 0)
                    adv_loss = adv_loss + nn.functional.binary_cross_entropy(adv_out, adv_target)
                    

                del x1; del x2; del outs;
                gc.collect(); gc.collect()
                torch.cuda.empty_cache()
                
            print('test loss : ', test_loss / n_batches)
            if adversarial:
                print('adversarial loss : ', adv_loss)
                adversarial_losses['train'].append(adv_loss.detach().cpu().numpy())
                
            # register model if best test loss
            if current_test_loss < min_test_loss:
                min_test_loss = current_test_loss
                print('-- best model found at epoch %d !!'%epoch)
                if torch.cuda.is_available():
                    with torch.cuda.device_of(next(models[0].parameters())):
                        models[0].cpu(); models[1].cpu()
                        best_model = [copy.deepcopy(models[0].get_dict(loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets])),
                                      copy.deepcopy(models[1].get_dict(loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets]))]
                        models[0].cuda(); models[1].cuda()
                else:
                    best_model = [copy.deepcopy(models[0].get_dict(loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets])),
                                  copy.deepcopy(models[1].get_dict(loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets]))]
            gc.collect(); gc.collect()
            torch.cuda.empty_cache()
                    
            # schedule training
            models[0].schedule(test_loss); models[1].schedule(test_loss)
            
            if epoch % plot_epochs == 0:
                plot_tf_reconstructions(datasets, meta_datasets, models, random_mode = random_mode, out=figures_folder+'/reconstructions_%d'%epoch, zero_extra_class=zero_extra_class)
                
            if epoch % save_epochs == 0:
                vae_1.save(results_folder + '/vae_1/%s_%d.t7'%(name,epoch), loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets], **save_with)
                vae_2.save(results_folder + '/vae_2/%s_%d.t7'%(name,epoch), loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets], **save_with)            
                if best_model:
                    torch.save({**save_with, **best_model[0]}, results_folder + '/vae_1/%s_best.t7'%(name))
                    torch.save({**save_with, **best_model[1]}, results_folder + '/vae_2/%s_best.t7'%(name))
                if adversarial:
                    torch.save(adversarial_losses, results_folder+'adversarial_log.t7')
                
        epoch += 1


#%% Graph functions
                
def plot_tf_reconstructions(datasets, meta_datasets, models, n_examples=3, out=None, random_mode='uniform', zero_extra_class=False):

    if random_mode == 'uniform':
        random_weights = np.random.random((len(datasets), n_examples))
    elif random_mode == 'constant':
        random_weights = np.ones((len(datasets), n_examples))
    elif random_mode == 'bernoulli':
        random_weights = torch.distributions.Bernoulli(torch.full((len(datasets), n_examples),0.5)).sample().numpy()

    ids = []; datas = []; metas = [];
    for i, d in enumerate(datasets):
        ids.append( np.random.permutation(d.data.shape[0])[:n_examples] )
        datas.append(d.data[ids[-1]])
        if issubclass(type(meta_datasets[i].data), list): 
            current_metadata = []
            for m_tmp in meta_datasets[i].data:
                m_tmp = m_tmp[ids[-1]]
                if zero_extra_class:
                    m_tmp = np.concatenate((m_tmp, np.zeros((m_tmp.shape[0], 1))), 1)
                    m_tmp[np.where(random_weights[i] == 0)] = np.array([0.]*(m_tmp.shape[1]-1)+[1])
                current_metadata.append(m_tmp)
            metas.extend(current_metadata)
        else:
            current_metadata = meta_datasets[i].data[ids[-1]]
            if zero_extra_class:
                current_metadata = np.concatenate((m_tmp, np.zeros((m_tmp.shape[0], 1))), 1)
                current_metadata[np.where(random_weights[i] == 0)] = np.array([0.]*(current_metadata.shape[1]-1)+[1])
            metas.append(current_metadata)
        
        
    datas = np.array(datas); 
    mixtures = np.sum( np.expand_dims(random_weights, -1) * datas, 0)
    
    # make forward passes
    signal_out = models[0].forward(mixtures)
    symbol_out = models[1].forward(metas)
    
    # compute transfers
    # WARNING here the conv module should work without *[0]
    signal_tf = models[0].decode(symbol_out["z_enc"][0])
    symbol_tf = models[1].decode(signal_out["z_enc"][0])
#    pdb.set_trace()
    n_symbols = len(models[1].pinput)
    grid = plt.GridSpec(n_examples*2, n_symbols * 3 )
    fig = plt.figure(figsize=(14,8))
    
    for i in range(n_examples):
        # plot original signal
        ax1 = fig.add_subplot(grid[2*i, :n_symbols])
        ax1.plot(mixtures[i])
        for j in range(len(datas)):
#            current_ax = fig.add_subplot(grid[2*i, j])
            ax1.plot(datas[j][i], linewidth=0.3)
        # plot original symbols
        for l in range(n_symbols):
            current_ax = fig.add_subplot(grid[2*i+1, l])
            current_ax.plot(metas[l][i].detach().cpu().numpy())
        # reconstructed signals
        ax2 = fig.add_subplot(grid[2*i, n_symbols:2*n_symbols])
        ax2.plot(mixtures[i], linewidth = 0.3)
        ax2.plot(signal_out['x_params'][0][i].cpu().detach().numpy())
        
        # reconstructed labels
        for l in range(n_symbols):
            current_ax = fig.add_subplot(grid[2*i+1, n_symbols+l])
            current_ax.plot(metas[l][i].detach().cpu().numpy(), linewidth = 0.3)
            current_ax.plot(symbol_out['x_params'][l][0][i].cpu().detach().numpy())
            
        # transferred data
        ax3 = fig.add_subplot(grid[2*i+1, 2*n_symbols:])
        ax3.plot(signal_tf[0]['out_params'][0][i].cpu().detach().numpy())
        ax3.plot(mixtures[i], linewidth = 0.3)
        for l in range(n_symbols):
            current_ax = fig.add_subplot(grid[2*i, 2*n_symbols+l])
            current_ax.plot(metas[l][i].detach().cpu().numpy(), linewidth = 0.3)
            current_ax.plot(symbol_tf[0]['out_params'][l][0][i].cpu().detach().numpy())
        
        
    if out:
        fig.savefig(out+'.pdf', format='pdf')

    del signal_out; del symbol_out
    torch.cuda.empty_cache()
    gc.collect(); gc.collect();

    return [fig], fig.axes
  
#%% Main


audioSet = DatasetAudio.load(args.dbroot)
preprocessing = Magnitude(audioSet.data, 'log', normalize=True)

# Extract sub-datasets
datasets = []; 
instrument_ids = [audioSet.classes['instrument'][d] for d in args.instruments]


for iid in instrument_ids:
    new_dataset = audioSet[np.where(audioSet.metadata['instrument'] == iid)[0]]
    if len(args.frames) == 0:
        print('taking the whole dataset...')
        new_dataset.flattenData(lambda x: x[:])
    elif len(args.frames)==2:
        print('taking between %d and %d...'%(args.frames[0], args.frames[1]))
        new_dataset.flattenData(lambda x: x[args.frames[0]:args.frames[1]])
    elif  len(args.frames)==1:
        print('taking frame %d'%(args.frames[0]))
        new_dataset.flattenData(lambda x: x[args.frames[0]])
    new_dataset.data = preprocessing(new_dataset.data)
    new_dataset.constructPartition([], ['train', 'test'], [0.8, 0.2], False)
    datasets.append(new_dataset)


#%% Extract metadata_dataset

print('making label dataset from audio dataset...')

meta_datasets = []
for d in range(len(datasets)):
    audioSetMeta = copy.deepcopy(audioSet)
    if args.label_type == 'binary':
        audioSetMeta.data = []
        for i, _ in enumerate(datasets[d].data):
            current_metadata = []
            for l in args.labels:
                current_metadata.append(oh.oneHot(datasets[d].metadata[l][i], audioSet.classes[l]['_length']))
            current_metadata = np.concatenate(current_metadata, 1)
            audioSetMeta.data.append(current_metadata)
        audioSetMeta.data = np.concatenate(audioSetMeta.data, 0)
    elif args.label_type == 'categorical':
        audioSetMeta.data = [[], [], []]
        for i, _ in enumerate(datasets[d].data):
            for j,l in enumerate(args.labels):
                audioSetMeta.data[j].append(oh.oneHot(datasets[d].metadata[l][i], audioSet.classes[l]['_length']))
        audioSetMeta.data = [np.concatenate(m, 0) for m in audioSetMeta.data]
    meta_datasets.append(audioSetMeta)



#%% Creating VAEs

# creating first vae
if args.load:
    print('loading vae...')
    model_name = args.load.split('/')[-1]
    model_path_1 = args.load+'/%s_0_%s.t7'%(model_name, args.load_epoch)
    model_path_2 = args.load+'/%s_1_%s.t7'%(model_name, args.load_epoch) 
    map_location = 'cpu' if args.cuda < 0 else 'cuda:%d'%args.cuda
    loaded = torch.load(model_path_1, map_location=map_location)
    vae_1 = loaded['class'].load(loaded, with_optimizer=True)
    loaded = torch.load(model_path_2, map_location=map_location)
    vae_2 = loaded['class'].load(loaded, with_optimizer=True)
    first_epoch = loaded['epoch']
    loss = loaded['loss']    
    preprocessing =loaded.get('preprocessing')
    if not loaded.get('label_type') is None:
        args.label_type = loaded['label_type']
else:
    print('building signal vae...')
    input_params = {'dim':datasets[0].data.shape[1], 'dist':dist.Normal}
    latent_params = []; hidden_params=[]
    for l in range(len(args.dims)):
        latent_params.append({'dim':args.dims[l], 'dist':dist.Normal})
        hidden_params.append({'dim':args.hidden_dims_1[l], 'nlayers':args.hidden_num_1[l], 'batch_norm':'batch', 'residual':True})
    vae_1 = VanillaVAE(input_params, latent_params, hidden_params, device=args.cuda)
    optim_params = {'optimizer':'Adam', 'optimArgs':{'lr':1e-3}, 'scheduler':'ReduceLROnPlateau'}
    vae_1.init_optimizer(optim_params)
    
    adversarial = None
    if args.adversarial:
        print('building adversarial network...')
        adversarial = AdversarialLayer(input_params, {'nlayers':args.adv_num, 'dim':args.adv_dim})
        adversarial.init_optimizer({'lr':args.adv_lr})

    print('building symbolic vae...')
    input_params = []
    for d in range(len(meta_datasets)):
        if args.label_type == 'binary':
            input_params.append({'dim':meta_datasets[d].data.shape[1], 'dist':dist.Bernoulli})
        elif args.label_type == 'categorical':
            input_params.extend([{'dim':9, 'dist':dist.Categorical}, {'dim':12, 'dist':dist.Categorical}, {'dim':5, 'dist':dist.Categorical}])

    if args.zero_extra_class:
        for i in range(len(input_params)):
            input_params[i]['dim'] += 1
        
    latent_params = []; hidden_params=[]
    for l in range(len(args.dims)):
        latent_params.append({'dim':args.dims[l], 'dist':dist.Normal})
        hidden_params.append({'dim':args.hidden_dims_2[l], 'nlayers':args.hidden_num_2[l], 'batch_norm':'batch', 'residual':True})
    vae_2 = VanillaVAE(input_params, latent_params, hidden_params, device=args.cuda)
    optim_params = {'optimizer':'Adam', 'optimArgs':{'lr':1e-3}, 'scheduler':'ReduceLROnPlateau'}
    vae_2.init_optimizer(optim_params)
    
    first_epoch = 0

if args.cuda >= 0:
    vae_1.cuda(args.cuda)
    vae_2.cuda(args.cuda)
    if args.adversarial:
        adversarial.cuda(args.cuda)

def stringify(l, separator="_"):
    s = ""
    for i in l:
        s += separator + str(i)
    return s

#%% Set up training

loss = SCANLoss({'beta':[args.beta_1, args.beta_2], 'cross_factors':[args.cross_1, args.cross_2], 'distance':args.regularization_type})

train_name = args.name+'_%s'%stringify(args.instruments)
train_options = {'epochs':args.epochs, 'save_epochs':args.save_epochs, 'name':args.name, 'results_folder':args.savedir + '/' + train_name, 
        'plot_epochs':args.plot_epochs, 'first_epoch':first_epoch, 'random_mode':args.random_mode, 'zero_extra_class':args.zero_extra_class}

plot_options = {}


#%% Train!
print('train!')
train_model(datasets, meta_datasets, [vae_1, vae_2], loss, preprocessing=[preprocessing, None], options=train_options, plot_options=plot_options, 
            adversarial=adversarial, save_with={'preprocessing':preprocessing, 'adversarial':adversarial, 'script_args':args})
    
