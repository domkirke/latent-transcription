#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:55:21 2019

@author: chemla
"""
import argparse, os, bisect, sys, pdb
import numpy as np
import torch
import matplotlib.pyplot as plt
import music21
import librosa
from skimage.transform import resize
from scipy import ndimage

import lt
from lt.data.audio import computeTransform
import lt.modules, lt.data
import lt.monitor.synthesize_audio as audio
from lt.criterions.criterion_scan import SCANLoss


#import data.signal.transforms as transforms
parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, choices=['info', 'midi', 'audio', 'sequence', 'morphing', 'trajectory', 'free_navigation'], help='generation mode')
parser.add_argument('-m', '--models', type=str, nargs='*', help="path of generation model")
parser.add_argument('-i', "--input", type=str, nargs = '*', default="./scan_audio", help="incoming files")
parser.add_argument('-o', "--output", type=str, default=None, help="destination folder")
parser.add_argument('-t', "--transposition", type=int, default=0, help="transpostion")
parser.add_argument('-s', "--latent_sample", type=int, default=1, help="latent sampling for 'alive' sound")
parser.add_argument('-c', "--cuda", type=str, default='cpu', help="cuda device")
parser.add_argument('-g', "--graphs", type=str, default=1, help="activates graph")
parser.add_argument('-d', "--dynamics", type=int, default=None, help="overwrites all dynamics")

parser.add_argument('--n_steps', type=int, default = 500, help="number of steps for various generations")
parser.add_argument('--n_generations', type=int, default = 10, help="number of steps for various generations")
parser.add_argument('--griffinlim_iters', type=int, default = 30, help="number of steps for various generations")
parser.add_argument('--interp_order', type=int, default = 1, help="order of interpolation")
parser.add_argument("--info", type=str, default=0, help="prints information about the corresponding mode")
parser.add_argument("--traj", type=str, default=["line"], nargs="*")

args = parser.parse_args()

sys.modules['models']=lt
sys.modules['models.vaes']=lt.modules
sys.modules['models.vaes.vae_vanillaVAE'] = lt.modules.modules_vanillaVAE
sys.modules['data'] = lt.data



#%% Useful classes

class Logger(object):
    def __init__(self, file):
        self.file = open(file, 'w+')
        
    def close(self):
        self.file.close()
        
    def __call__(self, *args):
        print(*args)
        self.file.write('\n')
        for a in args:
            self.file.write('%s '%a)


#%% Useful functions

def load_model(path, device='cpu'):
    # just load a model
    filename = os.path.basename(path).split('__')[0]+'_best'
    signal_data = torch.load(path+'/vae_1/%s.t7'%filename, map_location = args.cuda)
    symbol_data = torch.load(path+'/vae_2/%s.t7'%filename, map_location = args.cuda)
    signal_vae = signal_data['class'].load(signal_data)
    symbol_vae = symbol_data['class'].load(symbol_data)
    return [signal_vae, symbol_vae], [signal_data, symbol_vae]


def log(path, output, cuda='cpu'):
    print('Loading model...')
    models, model_dicts = load_model(path, cuda)
    if output is None:
        output = path
    logger = Logger(output+'/log.txt')
    
    logger('[Training_arguments]')
    for attr in filter(lambda x: x[0] != '_', dir(model_dicts[0]['script_args'])):
        logger(attr, ':', getattr(model_dicts[0]['script_args'], attr))
        
    logger('[Signal VAE properties]')
    logger('input parameters : ', models[0].pinput)
    logger('hidden parameters : ', models[0].phidden)
    logger('latent parameters : ', models[0].platent)
    logger('[Symbol VAE properties]')
    logger('input parameters : ', models[1].pinput)
    logger('hidden parameters : ', models[1].phidden)
    logger('latent parameters : ', models[1].platent)
    
    if os.path.isfile(path+'/final_errors.t7'):
        errors = torch.load(path+'/final_errors.t7')
        logger('[Model errors]')
        logger('\tOriginal errors : ')
        logger('\tSignal errors : ', errors['rec_errors'])
        logger('\tSymbolic errors : ')
        logger('\t\t log-likelihood :', errors['symbol_errors']['symbolic-ll'])
        logger('\t\t classification ratios :', errors['symbol_errors']['classif_ratio'])
        logger('\t\t confusion matrix :', errors['symbol_errors']['confusion'])

        logger('\t Transfer errors : ')
        logger('\tSignal errors : ', errors['rec_errors_tf'])
        logger('\tSymbolic errors : ')
        logger('\t\t log-likelihood :', errors['symbol_errors_tf']['symbolic-ll'])
        logger('\t\t classification ratios :', errors['symbol_errors_tf']['classif_ratio'])
        logger('\t\t confusion matrix :', errors['symbol_errors_tf']['confusion'])
        
    logger.close()


#%% MIDI import functions


def chord2onehot(chords, label_params, mode='separated_octave', dynamics=None):
    metadata = [np.zeros((meta_len['dim'])) for meta_len in label_params]
    if len(chords) > 0:
        for c in chords[0]:
            octave = c.octave
            pitch_class = (c.pitchClass + 3) % 12
            metadata[1][pitch_class] = 1
            metadata[0][octave] = 1
        if args.dynamics:
            dynamics = dynamics
        else:
            dynamics = min(chords[1] // ( 127 // metadata[2].shape[0]), metadata[2].shape[0]-1)
        metadata[2][dynamics] = 1
        
    return metadata
    
def get_velocity_from_chord(chord, callback=min):
    velocities = [n.volume.velocity for n in chord]
    return callback(velocities)
    
def midi2metadata(file_path, label_params, intervals = 0.02, transp=0, cut=True, keep_silences=False, keep_multipitch=False, dynamics=None):    
    mf = music21.converter.parseFile(file_path)
    mf = mf.chordify().flat
    
    orderedTimeList = []
    orderedChordList = []
    
    for i in mf.secondsMap:
        element = i['element']
        if type(element) == music21.chord.Chord:
            orderedTimeList.append(i['offsetSeconds'])
            element = element.transpose(transp)
            current_pitches = element.pitches
            current_velocity = get_velocity_from_chord(element)
            orderedChordList.append([current_pitches, current_velocity])
        elif type(element) == music21.note.Rest:
            orderedTimeList.append(i['offsetSeconds'])
            orderedChordList.append([])
            
    orderedChordList.append([])
    orderedTimeList.append(orderedTimeList[-1] + mf[-1].seconds)
    total_length = orderedTimeList[-1]
    intervals = np.arange(0, total_length, intervals)
            
    metadatas = []
    silence_mask = np.zeros_like(intervals);
    for i, off in enumerate(intervals):
        idx = max(bisect.bisect_left(orderedTimeList, off) - 1, 0)
        if idx < len(orderedChordList):
            metadatas.append(chord2onehot(orderedChordList[idx], label_params, dynamics=dynamics))
            active_pitches = np.where(metadatas[-1][0] == 1.)[0]
            if (orderedChordList[idx] != [] or keep_silences) and (len(active_pitches) == 1 or keep_multipitch):
                silence_mask[i] = 1
        else:
            metadatas.append([np.zeros((meta_len['dim'])) for meta_len in label_params])
            if keep_silences:
                silence_mask[i] = 1
            
    if cut:
        meta_dict = {'octave':filter_silence(np.array([x[0] for x in metadatas]), silence_mask), 
                     'pitch':filter_silence(np.array([x[1] for x in metadatas]), silence_mask),
                     'dynamics':filter_silence(np.array([x[2] for x in metadatas]), silence_mask)}
    else:
        meta_dict = {'octave':np.array([x[0] for x in metadatas]), 
                     'pitch':np.array([x[1] for x in metadatas]), 
                     'dynamics':np.array([x[2] for x in metadatas])}
    return meta_dict, silence_mask.astype(np.int)

def midi2audio(model_path, midi_paths, device='cpu', out=None, gl_iter = 30, transp=0, dynamics=None):
    def midi2audio_single(models, model_dicts, midi_path):
        # load MIDI file
        preprocessing = model_dicts[0]['preprocessing']
        label_params = models[1].pinput
        labels = model_dicts[0]['script_args'].labels
        metadata, silence_mask = midi2metadata(midi_path, models[1].pinput, keep_silences=False, keep_multipitch=True, cut=False, transp=transp, dynamics=dynamics)
        metadata = [metadata[l] for l in labels]
        
        # forward
        with torch.no_grad():
            metadata = models[1].format_input_data(metadata)
#            pdb.set_trace()
            symbol_latent = models[1].encode(metadata)
            if args.latent_sample:
                current_z  = models[1].platent[-1]['dist'](*symbol_latent[-1]['out_params']).sample()
            else:
                current_z = models[1].platent[-1]['dist'](*symbol_latent[-1]['out_params']).mean
            signal_out = models[0].decode(current_z)[0]['out_params'][0]
            
        # output graph
        if out is None:
            args.output = model_path + '/midi_reconstructions'
        filename = os.path.splitext(os.path.basename(midi_path))[0]
        
        if not os.path.isdir(args.output) and args.output != '':
            os.makedirs(args.output)
        if args.graphs:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax1.imshow(metadata[1].t(), aspect='auto')
            ax2 = fig.add_subplot(2,1,2)
            ax2.imshow(signal_out.cpu().detach().numpy().T, aspect='auto')
            fig.savefig(args.output+'/%s.pdf'%filename)
        
        # output signal
        signal_out = preprocessing.invert(signal_out)
        signal_out[np.where(silence_mask == 0.)[0]] = 0
        resynthesize(signal_out, upsampleFactor=4, out=args.output+'/%s.wav'%filename, gl_iter=gl_iter) 
        
    # load model
    models, model_dicts = load_model(model_path, device)
    models[0].eval(); models[1].eval()
    for midi_path in midi_paths:
        print('generating file %s...'%midi_path)
        midi2audio_single(models, model_dicts, midi_path)
    


def filter_silence(metadata, mask):
    if issubclass(type(metadata), list):
        metadata = [m[np.where(mask != 0)] for m in metadata]
    else:
        metadata = metadata[np.where(mask != 0)]
    return metadata

#%% NSGT inversion methods 
    
def resynthesize(x_in, grain_len = 6.0, upsampleFactor = 10, out=None, preprocessing=None, threshold = 1e-4, normalize=True, gl_iter=30):
    NSGT_LENGTHS = {0.5:157, 1.0:313, 2.0:626}
    target_len = int(grain_len * 22050)
    nsgt_chunk_size = int(np.ceil(grain_len*313))
        
    # upsample incoming transform
    x_in = x_in.cpu().detach().numpy()
    current_max = np.amax(x_in)
    x_in = resize(x_in/current_max, (int(x_in.shape[0]*upsampleFactor), x_in.shape[1]))
    x_in = x_in * current_max
    
    # invert preprocessing
    if preprocessing: 
        x_in = preprocessing.invert(x_in)
    x_in[x_in < threshold] = 0
    
    # fill original spectra with zeros
    original_length = x_in.shape[0]
    padded_length = (original_length // nsgt_chunk_size + 1) * nsgt_chunk_size 
    x_padded = np.zeros((padded_length, x_in.shape[1]))
    x_padded[:original_length] = x_in
    
    # split in chunks and convert
    chunks = np.split(x_padded, padded_length // nsgt_chunk_size)
    audio_chunks = [None]*len(chunks)
    for i, chunk in enumerate(chunks):
        chunk_resynth = audio.regenerateAudioNSGT(chunk.T, 30, 11000, 48, iterations = gl_iter, targetLen = target_len, testSize=True)
        audio_chunks[i] = chunk_resynth
        
    final_signal = np.concatenate(audio_chunks)
    final_signal = librosa.util.normalize(final_signal)
    # normalize    
    print('exporting at ...', out+'.wav')
    if out:
        librosa.output.write_wav(out+'.wav', final_signal, 22050)
        
    return final_signal


def audio2model(model_path, audio_paths, out=args.output, sample=False, gl_iter = 30):
    def audio2model_single(model, model_dicts, audio_path):
        preprocessing = model_dicts[0]['preprocessing']
        
        # load audio file
        options = {'minFreq':30, 'maxFreq':11000, 'nsgtBins':48, 'resampleTo':11000, 'downsampleFactor':10}
        transform = computeTransform([audio_path], 'nsgt-cqt', options={})[0]
        transform = preprocessing(transform)
            
        # forward
        with torch.no_grad():
            signal_in = model[0].format_input_data(transform)
            vae_out = model[0].forward(signal_in, sample=sample)
            # signal output
            signal_out = vae_out['x_params'][0]
            # symbol output
            if args.latent_sample:
                z = vae_out['z_enc'][-1]
            else:
                z = vae_out['z_params_enc'][-1][0]
                
            symbol_out = model[1].decode(z, sample=sample)
            
        
        # output graph
        if out is None:
            args.output = model_path + '/audio_reconstructions'
        if not os.path.isdir(args.output) and args.output != '':
            os.makedirs(args.output)
        filename = os.path.splitext(os.path.basename(audio_path))[0]
            
        if args.graphs:
            fig = plt.figure()
            ax1 = fig.add_subplot(3,1,1)
            ax1.imshow(transform.T, aspect='auto')
            ax2 = fig.add_subplot(3,1,2)
            ax2.imshow(signal_out.cpu().detach().numpy().T, aspect='auto')
            ax3 = fig.add_subplot(3,1,3)
            ax3.imshow(symbol_out[0]['out_params'][0][0].cpu().detach().numpy(), aspect='auto')
            fig.savefig(args.output+'/%s.pdf'%filename)
        
        # output signal
        signal_out = preprocessing.invert(signal_out)
        resynthesize(signal_out, upsampleFactor=1, out=args.output+'/%s.wav'%filename, gl_iter = gl_iter) 
          
    # load model
    models, model_dicts = load_model(model_path, args.cuda)
    models[0].eval(); models[1].eval()
    
    for audio_path in audio_paths:
        print('generating file %s...'%audio_path)
        audio2model_single(models, model_dicts, audio_path)    
        
#%%
        
        
def path2audio(model, current_z, n_interp=1, order_interp = 1, out=None, preprocessing=None, graphs=True, gl_iter = 30):
    
    # get corresponding sound distribution
    model.eval()
    vae_out = model.decode( model.format_input_data(current_z), sample=False )
    
    signal_out = vae_out[0]['out_params'][0].detach()
        
    if graphs:
        fig = plt.figure()
        plt.imshow(signal_out.cpu().detach().numpy(), aspect='auto')
        fig.savefig(out)
    
    # output signal
    if not preprocessing is None:
        signal_out = preprocessing.invert(signal_out)
    resynthesize(signal_out, upsampleFactor=1, out=out, gl_iter = gl_iter)      



def sequence2audio(model_path, sequence, n_interp=1, order_interp = 1, out=None,  preprocessing=None, device='cpu', graphs=True, gl_iter = 30):
    # load model
    sequence = [int(i) for i in sequence]
    models, model_dicts = load_model(model_path, args.cuda)
    preprocessing = model_dicts[0]['preprocessing']
    assert len(sequence) != 0 and not len(sequence) % len(models[1].pinput)
    labels = np.split(np.array(sequence), len(sequence) // len(models[1].pinput))

    models[0].eval(); models[1].eval()
    metadata = [ np.zeros((len(labels), o['dim'])) for o in models[1].pinput ]
    for i in range(len(labels)):
        for j in range(len(models[1].pinput)):
            metadata[j][i, labels[i][j]] = 1
    

    # get corresponding distributions in latent space
    with torch.no_grad():
        symbol_latent = models[1].encode(models[1].format_input_data(metadata))
        
    if order_interp == 0:
        symbol_latent = tuple([  torch.cat([ m[np.newaxis, i].repeat(n_interp, 1) for i in range(m.shape[0]) ], 0)  for m in symbol_latent[-1]['out_params'] ])
        if args.latent_sample:
            current_z  = models[1].platent[-1]['dist'](*symbol_latent).sample()
        else:
            current_z = models[1].platent[-1]['dist'](*symbol_latent).mean
    else:
        symbol_latent = models[1].platent[-1]['dist'](*symbol_latent[-1]['out_params']).mean
        coord_interp = np.linspace(0, symbol_latent.shape[0]-1, (symbol_latent.shape[0]-1)*n_interp)
        current_z = np.zeros((len(coord_interp), symbol_latent.shape[1]))
        for i,y in enumerate(coord_interp):
            current_z[i] = ndimage.map_coordinates(symbol_latent, [y * np.ones(symbol_latent.shape[1]), np.arange(symbol_latent.shape[1])], order=order_interp)
        current_z = torch.from_numpy(current_z)
        
        
        # make interpolation

        
    if out is None:
        if order_interp == 0:
            out = model_path+'/sequences'
        else:
            out = model_path+'/interp_%d'%order_interp
    if not os.path.isdir(out) and args.output != '':
        os.makedirs(out)
    filename = ''
    for i in sequence:
        filename += str(i)
    filename = 'sequence_'+filename
    path2audio(models[0], current_z, n_interp=n_interp, order_interp=order_interp, preprocessing=preprocessing, graphs=graphs, out=out+'/'+filename, gl_iter = gl_iter)
    

    
def get_trajectory(trajectory_type, z_dim, n_trajectories=1, n_steps=1000):
    trajectories = []
    if trajectory_type == "line":
        for i in range(n_trajectories):
            origins = np.random.multivariate_normal(np.zeros((z_dim)), np.diag(3*np.ones((z_dim))), 2)
            coord_interp = np.linspace(0, 1, n_steps)
            z_interp = np.zeros((len(coord_interp), origins.shape[1]))
            for i,y in enumerate(coord_interp):
                z_interp[i] = ndimage.map_coordinates(origins, [y * np.ones(origins.shape[1]), np.arange(origins.shape[1])], order=2)
            z_traj = torch.from_numpy(z_interp)
            trajectories.append(z_traj)
            
    elif trajectory_type == "circle":
        for i in range(n_trajectories):
            origin = np.random.random(z_dim)*4 - 2
#            pdb.set_trace()
            radius = np.random.random() * 8 # draw a random radius
            angles = np.linspace(0, np.pi, n_steps)
            angles[-1] *= 2
            angles = np.repeat(angles[np.newaxis, :], z_dim - 1, axis=0) # nd-circle is defined by z_dim-1 angles
            angles = angles - np.random.random(z_dim-1)[:, np.newaxis]*2*np.pi # dephase everything
            z_traj = np.zeros((n_steps, z_dim))
            for i in range(z_dim-1):
                if i == 0:
                    z_traj[:, i] = np.cos(angles[i, :])
                else:
                    z_traj[:, i] = np.multiply( np.cumprod(np.sin(angles[:i, :]), axis=0)[-1] , np.cos(angles[i, :]) )
            z_traj[:, -1] = np.cumprod(np.sin(angles), axis=0)[-1]
            z_traj = z_traj * radius
            z_traj = z_traj + origin
            trajectories.append(z_traj)
            
    elif trajectory_type == "helix":
        for i in range(n_trajectories):
            origin = np.random.random(z_dim)*4 - 2
#            pdb.set_trace()
            radius = np.random.random(2) * 8 # draw a random radius
            radius = np.linspace(radius[0], radius[1], n_steps)
            angles = np.linspace(0, np.pi, n_steps)
            angles[-1] *= 2
            angles = np.repeat(angles[np.newaxis, :], z_dim - 1, axis=0) # nd-circle is defined by z_dim-1 angles
            angles = angles - np.random.random(z_dim-1)[:, np.newaxis]*2*np.pi # dephase everything
            z_traj = np.zeros((n_steps, z_dim))
            for i in range(z_dim-1):
                if i == 0:
                    z_traj[:, i] = np.cos(angles[i, :])
                else:
                    z_traj[:, i] = np.multiply( np.cumprod(np.sin(angles[:i, :]), axis=0)[-1] , np.cos(angles[i, :]) )
            z_traj[:, -1] = np.cumprod(np.sin(angles), axis=0)[-1]
            z_traj = z_traj * radius[:, np.newaxis] + origin
            trajectories.append(z_traj)
    return trajectories
 
def trajectory2audio(model_path, traj_types, n_trajectories=1, n_steps=1000, order_interp = 1, out=None,  preprocessing=None, device='cpu', graphs=True, gl_iter = 30):
    # load model
    models, model_dicts = load_model(model_path, args.cuda)
    preprocessing = model_dicts[0]['preprocessing']
    
    if out is None:
        out = model_path
    print(out)
    if not os.path.isdir(out+'/trajectories'):
        os.makedirs(out+'/trajectories')
    # generate trajectories
    for traj_type in traj_types:
        trajectories = get_trajectory(traj_type, models[0].platent[-1]['dim'], n_trajectories, n_steps)
        # forward trajectory
        for i,t in enumerate(trajectories):
            path2audio(models[0], t, n_interp=1, preprocessing=preprocessing, out=out+'/trajectories/%s_%s'%(traj_type,  i), gl_iter = gl_iter)
        
        
#        z_latent = np.concatenate(trajectories, axis=0)
#        signal_out = models[0].decode(models[0].format_input_data(z_latent))
#        symbol_out = models[1].decode(models[1].format_input_data(z_latent))
#    
#        for i in range(len(trajectories_idx)-1):
#            current_signal = signal_out[0]['out_params'][0][trajectories_idx[i]:trajectories_idx[i+1]]
#            print(current_signal.shape)
#    
        
    


    
#%%
    
plt.ioff()

if __name__ == "__main__":
    
    if args.mode == 'info':
        # just print information about a model
        if args.info:
            print("just outputs information about the selected model.\n Required inputs : model")
        
        for model in args.models:
            log(model, None)
        
            
    elif args.mode == 'midi':
        # takes a midi file and pass it into the encoder
        if args.info:
            print("takes a MIDI file to feed the model. Must have a number of tracks equal to the number of instruments in the model.\n Required inputs : -i input midi file -o destination output -t transposition -s sampling")
            
        for model_path in args.models:
            print('MODEL : %s'%model_path)
            midi2audio(model_path, args.input, out=args.output, device=args.cuda, gl_iter = args.griffinlim_iters, transp=args.transposition, dynamics=args.dynamics)
        
    
    elif args.mode == 'audio':
        # takes an audio file and pass into the encoder
        if args.info:
            print("feed the system with an audio file and get the reconstruction and corresponding inferred symbols.")
            
        for model_path in args.models:
            print('MODEL : %s'%model_path)
            audio2model(model_path, args.input, out=args.output, gl_iter = args.griffinlim_iters)        
            
    
    elif args.mode == 'sequence':
        if args.info:
            print("Directly output the corresponding sounds for a given sequence of labels")
            
        for model_path in args.models:
            print('MODEL : %s'%model_path)
            sequence2audio(model_path, args.input, n_interp=args.n_steps, order_interp=0, out=args.output, device=args.cuda, gl_iter = args.griffinlim_iters)
            
    
    elif args.mode == 'morphing':
        if args.info:
            print("Slowly morphs between several labels labels")
            
        for model_path in args.models:
            print('MODEL : %s'%model_path)
            sequence2audio(model_path, args.input, n_interp=args.n_steps, order_interp=args.interp_order, out=args.output, device=args.cuda, gl_iter = args.griffinlim_iters)
            
    
    elif args.mode == 'trajectory':
        if args.info:
            print("Generate sound with a geometric trajectory ")
    
        for model_path in args.models:
            print('MODEL : %s'%model_path)
            trajectory2audio(model_path, args.traj, n_trajectories = args.n_generations, n_steps = args.n_steps, out=args.output, device=args.cuda, order_interp=args.interp_order, gl_iter = args.griffinlim_iters)


