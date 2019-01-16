#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:12:03 2019

@author: chemla
"""

import os, copy
import matplotlib.pyplot as plt
import numpy as np
import torch

import data.metadata as mc
from scan_poly import *
from data.audio import DatasetAudio
from utils.onehot import oneHot, fromOneHot
from matplotlib.gridspec import GridSpec


dataset_root = "/Users/chemla/Datasets/acidsInstruments-ordinario"
model = "saves/remote/scan_multi_32d_2000h__Flute"
out = "saves/remote"

#%%

def load_model(path, device='cpu'):
    # just load a model
    filename = os.path.basename(model).split('__')[0]+'_best'
    signal_data = torch.load(path+'/vae_1/%s.t7'%filename, map_location = 'cpu')
    symbol_data = torch.load(path+'/vae_2/%s.t7'%filename, map_location = 'cpu')
    signal_vae = signal_data['class'].load(signal_data)
    symbol_vae = symbol_data['class'].load(symbol_data)
    return [signal_vae, symbol_vae], [signal_data, symbol_vae]


def get_subdatasets(audioSet, args, partitions=None, preprocessing=None):
    datasets = []; 
    instrument_ids = [audioSet.classes['instrument'][d] for d in args.instruments]
    for n, iid in enumerate(instrument_ids):
        new_dataset = audioSet[np.where(audioSet.metadata['instrument'] == iid)[0]]
        new_dataset.importData(None, audioOptions)
        if len(args.frames) == 0:
            print('taking the whole dataset...')
            new_dataset.flattenData(lambda x: x[:])
        elif len(args.frames)==2:
            print('taking between %d and %d...'%(args.frames[0], args.frames[1]))
            new_dataset.flattenData(lambda x: x[args.frames[0]:args.frames[1]])
        elif  len(args.frames)==1:
            print('taking frame %d'%(args.frames[0]))
            new_dataset.flattenData(lambda x: x[args.frames[0]])
            
        new_dataset.partitions = partitions[n]
            
        if preprocessing:
            new_dataset.data = preprocessing(new_dataset.data)
            
        datasets.append(new_dataset)
        
    return datasets

def get_symbolic_datasets(audioSet, datasets, args):
    zero_extra_class = hasattr(args, 'zero_extra_class') and args.zero_extra_class    
    meta_datasets = []
    for d in range(len(datasets)):
        audioSetMeta = copy.deepcopy(audioSet)
        if args.label_type == 'binary':
            audioSetMeta.data = []
            for i, _ in enumerate(datasets[d].data):
                current_metadata = []
                for l in args.labels:
                    label_size = audioSet.classes[l]['_length']
                    current_metadata.append(oneHot(datasets[d].metadata[l][i], label_size))
                current_metadata = np.concatenate(current_metadata, 1)
                audioSetMeta.data.append(current_metadata)
            audioSetMeta.data = np.concatenate(audioSetMeta.data, 0)
        elif args.label_type == 'categorical':
            audioSetMeta.data = [[], [], []]
            for i, _ in enumerate(datasets[d].data):
                for j,l in enumerate(args.labels):
                    label_size = audioSet.classes[l]['_length']
                    if zero_extra_class:
                        label_size += 1
                    audioSetMeta.data[j].append(oneHot(datasets[d].metadata[l][i], label_size))
            audioSetMeta.data = [np.concatenate(m, 0) for m in audioSetMeta.data]
        meta_datasets.append(audioSetMeta)
    return meta_datasets


#%% Load dataset


audioOptions = {                      
  "dataPrefix":dataset_root,    
  "dataDirectory":dataset_root+'/data',
  "analysisDirectory":dataset_root+'/analysis/ordinario/nsgt-cqt',
  "transformName":"nsgt-cqt",                                
  "verbose":True,                                         
  "forceUpdate":False,
  'tasks':['instrument', 'family', 'dynamics','pitch', 'octave'],
  'taskCallback':[mc.importRawLabel]*3+[mc.importRawNumber]*2
};
        
print('[Info] Loading data...')

# Create dataset object
audioSet = DatasetAudio(audioOptions);

# Recursively check in the given directory for data
audioSet.listDirectory();
audioSet.importMetadataTasks()
audioSet.classes['instrument'] = { 'English-Horn':0, 'French-Horn':1, 'Tenor-Trombone':2, 'Trumpet-C':3,
                                    'Piano':4, 'Violin':5, 'Violoncello':6, 'Alto-Sax':7, 'Bassoon':8,
                                    'Clarinet-Bb':9, 'Flute':10, 'Oboe':11, '_length':12 }

audioSet.classes['pitch'] = {'A':0, 'A#':1, 'B':2, 'C':3, 'C#':4, 'D':5, 'D#':6, 'E':7, 'F':8, 'F#':9, 'G':10, 'G#':11, '_length':12}
audioSet.classes['octave'] = {str(i):i for i in range(9)}
audioSet.classes['octave']['_length'] = 9





#%% Solo example

fig = plt.figure(figsize=(14,5))
grid = GridSpec(6, 27)


spec_id = 30

vaes, dicts = load_model(model)
current_datasets = get_subdatasets(audioSet, dicts[0]['script_args'], preprocessing=None, partitions=dicts[0]['partitions'])
meta_datasets = get_symbolic_datasets(audioSet, current_datasets, dicts[0]['script_args'])



signal_vae = vaes[0]; symbol_vae = vaes[1];
signal_vae.eval(); symbol_vae.eval()

preprocessing = dicts[0]['preprocessing']
current_spec = preprocessing(current_datasets[0].data[spec_id])[np.newaxis, :]
current_symbols = [s[np.newaxis, spec_id] for s in meta_datasets[0].data]
#current_symbols = [current_symbols[i] for i in range(len(current_symbols))]

signal_out = signal_vae(current_spec)
symbol_out = symbol_vae(current_symbols)

signal_z = signal_out['z_params_enc'][0][0]
symbol_z = symbol_out['z_params_enc'][0][0]

signal_tf_out = signal_vae.decode(symbol_z)[0]['out_params'][0]
symbol_tf_out = symbol_vae.decode(signal_z)[0]['out_params']
symbol_tf_out = [s[0] for s in symbol_tf_out]


# PLOT

ax = fig.add_subplot(grid[0, :9])
ax.plot(current_spec[0], linewidth=1.0)
 #plt.yticks([],[])
plt.setp(ax.get_xticklabels(), visible=False); 
plt.setp(ax.get_yticklabels(), fontsize=6, x=3e-2); 
#ax.set_title("Original")


ax = fig.add_subplot(grid[0, 9:18])    
ax.plot(current_spec[0], linewidth=0.7)
ax.plot(signal_out['x_params'][0][0].detach().numpy(), linewidth=1.0)
plt.setp(ax.get_xticklabels(), visible=False);  plt.yticks([],[])
#ax.set_title("Reconstructions")

ax = fig.add_subplot(grid[1, 18:])    
ax.plot(current_spec[0], linewidth=0.7)
ax.plot(signal_tf_out[0].detach().numpy(), linewidth=1.0)
plt.setp(ax.get_xticklabels(), visible=False);  plt.yticks([],[])
#ax.set_title("Transfer")

for i in range(3):    
    ax = fig.add_subplot(grid[1, 3*i:3*(i+1)])
    ax.plot(current_symbols[i][0].detach().numpy())
    ax.set_xticks(list(range(symbol_vae.pinput[i]['dim'])))
    if i != 0:
        plt.setp(ax.get_yticklabels(), visible=False); 
    plt.setp(ax.get_yticklabels(), fontsize=6, x=3e-2); 
    ax.set_yticks([0,1]); plt.setp(ax.get_xticklabels(), visible=False); 
    
    ax = fig.add_subplot(grid[1, 3*i+9:3*(i+1)+9])
#    ax.plot(current_symbols[i][0].detach().numpy(), linewidth=0.7)
    ax.plot(symbol_out['x_params'][i][0][0].detach().numpy())
    ax.set_xticks(list(range(symbol_vae.pinput[i]['dim'])))
    plt.setp(ax.get_yticklabels(), visible=False); 
    ax.set_yticks([0,1]); plt.setp(ax.get_xticklabels(), visible=False); 
    
    ax = fig.add_subplot(grid[0, 3*i+18:3*(i+1)+18])
#    ax.plot(current_symbols[i][0].detach().numpy(), linewidth=0.7)
    ax.plot(symbol_tf_out[i][0].detach().numpy())
    ax.set_xticks(list(range(symbol_vae.pinput[i]['dim'])))
    plt.setp(ax.get_yticklabels(), visible=False); 
    ax.set_yticks([0,1]); plt.setp(ax.get_xticklabels(), visible=False); 
    
#for i in range(3):
#    ax = fig.add_subplot(grid[0, 7+i])
#    ax.plot()
#    
    
#%% Duo example 

model = "saves/remote/results_2inst/scan_multi_32d_5000h_800h__Alto-Sax_Violin"
spec_ids = [30, 100]

vaes, dicts = load_model(model)
current_datasets = get_subdatasets(audioSet, dicts[0]['script_args'], preprocessing=None, partitions=dicts[0]['partitions'])
meta_datasets = get_symbolic_datasets(audioSet, current_datasets, dicts[0]['script_args'])

signal_vae = vaes[0]; symbol_vae = vaes[1];
signal_vae.eval(); symbol_vae.eval()


preprocessing = dicts[0]['preprocessing']

#fig = plt.figure()
#plt.plot(preprocessing(current_datasets[0].data[spec_ids[0]]), linewidth=0.5)
#plt.plot(preprocessing(current_datasets[1].data[spec_ids[1]]), linewidth=0.5)
#plt.plot(preprocessing(sum([current_datasets[i].data[spec_ids[i]] for i in range(len(current_datasets))])))



current_spec = sum([current_datasets[i].data[spec_ids[i]] for i in range(len(current_datasets))])
current_spec = preprocessing(current_spec)[np.newaxis, :]

current_symbols = []
for i in range(len(current_datasets)):
    current_symbols.extend([s[np.newaxis, spec_ids[i]] for s in meta_datasets[i].data])
#current_symbols = [current_symbols[i] for i in range(len(current_symbols))]


signal_out = signal_vae(current_spec)
symbol_out = symbol_vae(current_symbols)

signal_z = signal_out['z_params_enc'][0][0]
symbol_z = symbol_out['z_params_enc'][0][0]

signal_tf_out = signal_vae.decode(symbol_z)[0]['out_params'][0]
symbol_tf_out = symbol_vae.decode(signal_z)[0]['out_params']
symbol_tf_out = [s[0] for s in symbol_tf_out]


# PLOT
ax = fig.add_subplot(grid[2, :9])
ax.plot(current_spec[0], linewidth=1.0)

spectrums = [current_datasets[i].data[spec_ids[i]] for i in range(len(current_datasets))]
for s in spectrums:
    ax.plot(preprocessing(s), linewidth=0.1)
 #plt.yticks([],[])
plt.setp(ax.get_xticklabels(), visible=False); 
plt.setp(ax.get_yticklabels(), fontsize=6, x=3e-2); 


ax = fig.add_subplot(grid[2, 9:18])    
ax.plot(current_spec[0], linewidth=0.5)
ax.plot(signal_out['x_params'][0][0].detach().numpy(), linewidth=0.8)
plt.setp(ax.get_xticklabels(), visible=False);  plt.yticks([],[])

ax = fig.add_subplot(grid[3, 18:])    
ax.plot(current_spec[0], linewidth=0.5)
ax.plot(signal_tf_out[0].detach().numpy(), linewidth=0.8)
plt.setp(ax.get_xticklabels(), visible=False);  plt.yticks([],[])
#
for i in range(len(symbol_tf_out)):    
    ax = fig.add_subplot(grid[3, 3*(i%3):3*((i%3)+1)])
    ax.plot(current_symbols[i][0].detach().numpy())
    ax.set_xticks(list(range(symbol_vae.pinput[i]['dim'])))
    if i != 0:
        plt.setp(ax.get_yticklabels(), visible=False); 
    plt.setp(ax.get_yticklabels(), fontsize=6, x=3e-2); 
    ax.set_yticks([0,1]); plt.setp(ax.get_xticklabels(), visible=False); 
    
    ax = fig.add_subplot(grid[3, 3*(i%3)+9:3*((i%3)+1)+9])
#    ax.plot(current_symbols[i][0].detach().numpy(), linewidth=0.7)
    ax.plot(symbol_out['x_params'][i][0][0].detach().numpy())
    ax.set_xticks(list(range(symbol_vae.pinput[i]['dim'])))
    plt.setp(ax.get_yticklabels(), visible=False); 
    ax.set_yticks([0,1]); plt.setp(ax.get_xticklabels(), visible=False); 
    
    ax = fig.add_subplot(grid[2, 3*(i%3)+18:3*((i%3)+1)+18])
#    ax.plot(current_symbols[i][0].detach().numpy(), linewidth=0.7)
    ax.plot(symbol_tf_out[i][0].detach().numpy())
    ax.set_xticks(list(range(symbol_vae.pinput[i]['dim'])))
    plt.setp(ax.get_yticklabels(), visible=False); 
    ax.set_yticks([0,1]); plt.setp(ax.get_xticklabels(), visible=False); 
#   

#%% Trio examples 

model = "saves/remote/scan_3/scan_multi_32d_3000h_1500h__Alto-Sax_Violin_Trumpet-C"
spec_ids = [30, 200, 113]

vaes, dicts = load_model(model)
current_datasets = get_subdatasets(audioSet, dicts[0]['script_args'], preprocessing=None, partitions=dicts[0]['partitions'])
meta_datasets = get_symbolic_datasets(audioSet, current_datasets, dicts[0]['script_args'])

signal_vae = vaes[0]; symbol_vae = vaes[1];
signal_vae.eval(); symbol_vae.eval()


preprocessing = dicts[0]['preprocessing']
current_spec = sum([current_datasets[i].data[spec_ids[i]] for i in range(len(current_datasets))])
current_spec = preprocessing(current_spec)[np.newaxis, :]

current_symbols = []
for i in range(len(current_datasets)):
    current_symbols.extend([s[np.newaxis, spec_ids[i]] for s in meta_datasets[i].data])
#current_symbols = [current_symbols[i] for i in range(len(current_symbols))]


signal_out = signal_vae(current_spec)
symbol_out = symbol_vae(current_symbols)

signal_z = signal_out['z_params_enc'][0][0]
symbol_z = symbol_out['z_params_enc'][0][0]

signal_tf_out = signal_vae.decode(symbol_z)[0]['out_params'][0]
symbol_tf_out = symbol_vae.decode(signal_z)[0]['out_params']
symbol_tf_out = [s[0] for s in symbol_tf_out]

# PLOT
spectrums = [current_datasets[i].data[spec_ids[i]] for i in range(len(current_datasets))]
ax = fig.add_subplot(grid[4, :9])
ax.plot(current_spec[0], linewidth=1.0)
for s in spectrums:
    ax.plot(preprocessing(s), linewidth=0.1)
 #plt.yticks([],[])
plt.setp(ax.get_xticklabels(), visible=False); 
plt.setp(ax.get_yticklabels(), fontsize=6, x=3e-2); 


ax = fig.add_subplot(grid[4, 9:18])    
ax.plot(current_spec[0], linewidth=0.5)
ax.plot(signal_out['x_params'][0][0].detach().numpy(), linewidth=0.8)
plt.setp(ax.get_xticklabels(), visible=False);  plt.yticks([],[])

ax = fig.add_subplot(grid[5, 18:])    
ax.plot(current_spec[0], linewidth=0.5)
ax.plot(signal_tf_out[0].detach().numpy(), linewidth=0.8)
plt.setp(ax.get_xticklabels(), visible=False);  plt.yticks([],[])
#
for i in range(len(symbol_tf_out)):    
    ax = fig.add_subplot(grid[5, 3*(i%3):3*((i%3)+1)])
    ax.plot(current_symbols[i][0].detach().numpy())
    ax.set_xticks(list(range(symbol_vae.pinput[i]['dim'])))
    if i != 0:
        plt.setp(ax.get_yticklabels(), visible=False); 
    plt.setp(ax.get_yticklabels(), fontsize=6, x=3e-2); 
    ax.set_yticks([0,1]); plt.setp(ax.get_xticklabels(), visible=False); 
    
    ax = fig.add_subplot(grid[5, 3*(i%3)+9:3*((i%3)+1)+9])
#    ax.plot(current_symbols[i][0].detach().numpy(), linewidth=0.7)
    ax.plot(symbol_out['x_params'][i][0][0].detach().numpy())
    ax.set_xticks(list(range(symbol_vae.pinput[i]['dim'])))
    plt.setp(ax.get_yticklabels(), visible=False); 
    ax.set_yticks([0,1]); plt.setp(ax.get_xticklabels(), visible=False); 
    
    ax = fig.add_subplot(grid[4, 3*(i%3)+18:3*((i%3)+1)+18])
#    ax.plot(current_symbols[i][0].detach().numpy(), linewidth=0.7)
    ax.plot(symbol_tf_out[i][0].detach().numpy())
    ax.set_xticks(list(range(symbol_vae.pinput[i]['dim'])))
    plt.setp(ax.get_yticklabels(), visible=False); 
    ax.set_yticks([0,1]); plt.setp(ax.get_xticklabels(), visible=False); 
#   

#%% SAVE
    
    
fig.savefig(out+'/final_reconstructions.svg', format='svg')    
    