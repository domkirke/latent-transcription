#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:12:40 2018

@author: chemla
"""

import sys
import torch
import numpy as np, bisect

from skimage.transform import resize
import music21, argparse, os, librosa
import lt
import lt.data.metadata as mc
from lt.data.audio import DatasetAudio
from lt.data.preprocessing import Magnitude
from lt.modules.modules_classifier import Classifier

import lt.monitor.synthesize_audio as audio


#%% Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, default='saves/scan_multi_32d_2000h__Flute')
parser.add_argument('--output', type=str, default=None)

parser.add_argument('--dataroot', type=str, default='/Users/chemla/Datasets/flute-audio-labelled-database-AMT/Recordings')
parser.add_argument('--midiroot', type=str, default='/Users/chemla/Datasets/flute-audio-labelled-database-AMT/Aligned_files/mid_files')

parser.add_argument('--transform', type=str, default='nsgt-cqt')    
parser.add_argument('--instruments', type=str, nargs =  '+', default = ['Flute'])

parser.add_argument('--cuda', type=int, default=-1)

parser.add_argument('--plot', type=int, default=1)    
parser.add_argument('--errors', type=int, default=1)    
parser.add_argument('--resynthesize', type=int, default=0)
parser.add_argument('--classifier', type=int, default=1)    

args = parser.parse_args()

if args.cuda < 0:
    args.cuda = 'cpu'

#%% Load model
        
# package translation hack
sys.modules['models']=lt
sys.modules['models.vaes']=lt.modules
sys.modules['models.vaes.vae_vanillaVAE'] = lt.modules.modules_vanillaVAE
sys.modules['data'] = lt.data

model_path = args.models
filename = os.path.basename(model_path).split('__')[0]+'_best'
signal_data = torch.load(model_path+'/vae_1/%s.t7'%filename, map_location = args.cuda)
symbol_data = torch.load(model_path+'/vae_2/%s.t7'%filename, map_location = args.cuda)
signal_vae = signal_data['class'].load(signal_data)
symbol_vae = symbol_data['class'].load(symbol_data)

script_args = signal_data['script_args']
preprocessing = signal_data['preprocessing']

if script_args.zero_extra_class:
    octave_len = 10
    pitch_len = 13
    dyn_len = 6
else:
    octave_len = 9
    pitch_len = 12
    dyn_len = 5

#%% MIDI import functions

def chord2onehot(chords, mode='separated_octave'):
    metadata = [np.zeros((pitch_len)), np.zeros((octave_len))]
    for c in chords:
        octave = c.octave
        pitch_class = (c.pitchClass + 3) % 12
        metadata[0][pitch_class] = 1
        metadata[1][octave] = 1
    return metadata
    
    
def midi2metadata(file_path, intervals, keep_silences=False, keep_multipitch=False):    
    mf = music21.converter.parseFile(file_path)
    mf = mf.chordify().flat
    
    orderedTimeList = []
    orderedChordList = []
    
    for i in mf.secondsMap:
        element = i['element']
        if type(element) == music21.chord.Chord:
            orderedTimeList.append(i['offsetSeconds'])
            orderedChordList.append(element.pitches)
        elif type(element) == music21.note.Rest:
            orderedTimeList.append(i['offsetSeconds'])
            orderedChordList.append([])
            
    orderedChordList.append([])
    orderedTimeList.append(orderedTimeList[-1] + mf[-1].seconds)
            
    metadatas = []
    silence_mask = np.zeros_like(intervals);
    for i, off in enumerate(intervals):
        idx = max(bisect.bisect_left(orderedTimeList, off) - 1, 0)
        if idx < len(orderedChordList):
            metadatas.append(chord2onehot(orderedChordList[idx]))
            active_pitches = np.where(metadatas[-1][0] == 1.)[0]
            if (orderedChordList[idx] != [] or keep_silences) and (len(active_pitches) == 1 or keep_multipitch):
                silence_mask[i] = 1
        else:
            metadatas.append([np.zeros((pitch_len)), np.zeros((octave_len)), np.zeros((dyn_len))])
            if keep_silences:
                silence_mask[i] = 1
            
                    
    meta_dict = {'pitch':filter_silence(np.array([x[0] for x in metadatas]), silence_mask), 
                 'octave':filter_silence(np.array([x[1] for x in metadatas]), silence_mask)}
    
    meta_dict['dynamics'] = np.zeros((meta_dict['pitch'].shape[0], dyn_len))
    meta_dict['dynamics'][:, 3] = 1
    
    return meta_dict, silence_mask.astype(np.int)



def filter_silence(metadata, mask):
    if issubclass(type(metadata), list):
        metadata = [m[np.where(mask != 0)] for m in metadata]
    else:
        metadata = metadata[np.where(mask != 0)]
    return metadata


#%% Flute dataset import 
    
# Import test dataset

testOptions = {                
  "dataDirectory":args.dataroot,
  "analysisDirectory":'coucou',
  "transformName":"nsgt-cqt",                                
  "verbose":False
};
        
print('[Info] Loading test data...')

# Create dataset object
testSet = DatasetAudio(testOptions);
testSet.listDirectory(check=False)


# create transforms if do not exists
transformList, transformParameters = testSet.getTransforms();
transformParameters['minFreq'] = 30
transformParameters['maxFreq'] = 11000
transformParameters['nsgtBins'] = 48
transformParameters['downsampleFactor'] = 10
testSet.computeTransforms(['nsgt-cqt'], [transformParameters], ['nsgt-cqt'], forceRecompute=False)
# import data
testSet.importData(None, testOptions)


for i,f in enumerate(testSet.files):
    filename = os.path.splitext(os.path.basename(f))[0]
    midi_name = args.midiroot+'/'+filename+'-lined.mid'
    # check file exists
    try:
        check = open(midi_name, 'r')
        check.close()
    except FileNotFoundError as e:
        # normalize nsgt
        preprocessing = Magnitude(testSet.data[i], 'log', normalize=True)
        print(e)
        continue
    
    # retrieve file duration
    y,sr = librosa.load(f)
    file_length = y.shape[0]/sr 
    
    # retrieve metadata
    intervals = np.linspace(0, file_length, testSet.data[i].shape[0])
    current_metadata, silence_mask = midi2metadata(midi_name, intervals, keep_silences=False, keep_multipitch=False)    
    testSet.data[i] = testSet.data[i][np.where(silence_mask != 0.)]
    
    for k in current_metadata.keys():
        if not k in testSet.metadata.keys():
            testSet.metadata[k] = [None]*len(testSet.files)
        testSet.metadata[k][i] = current_metadata[k]
        
#    print(testSet.data[i].shape, testSet.metadata['pitch'][i].shape)
        

testSet.classes = audioSet.classes
#%% Go!

import matplotlib.pyplot as plt
from skimage.transform import rescale
from lt_analyze import get_signal_errors, get_symbolic_errors


def plot_reconstructions(x_orig, x_reco, x_tf, out=None, log=True):
    fig = plt.figure()
    if not log:
        x_orig = x_orig.exp()
        x_reco = x_reco.exp()
        x_tf = x_tf.exp()
    x_orig = x_orig.cpu().detach().numpy()
    x_reco = x_reco.cpu().detach().numpy()
    x_tf = x_tf.cpu().detach().numpy()
    # original reconstructions
    ax1 = fig.add_subplot(131)
    ax1.imshow(x_orig.T, aspect='auto')
    ax2 = fig.add_subplot(132)
    ax2.imshow(x_reco.T, aspect='auto')
    ax3 = fig.add_subplot(133)
    ax3.imshow(x_tf.T, aspect='auto')

    if out:
        fig.savefig(out)
        
    return fig


def resynthesize(x_in, grain_len = 2.0, upsampleFactor = 10, out=None, preprocessing=None, threshold = 1e-4, normalize=True):
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
        chunk_resynth = audio.regenerateAudioNSGT(chunk.T, 30, 11000, 48, iterations = 30, targetLen = target_len, testSize=True)
        audio_chunks[i] = chunk_resynth
        
    final_signal = np.concatenate(audio_chunks)
    # normalize    
    if out:
        librosa.output.write_wav(out, final_signal, 22050)
        
    return final_signal

#%%

if args.output is None:
    args.output = args.models + '/flute_results'
results_folder = args.output
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)
if not os.path.isdir(results_folder+'/plots'):
    os.makedirs(results_folder+'/plots')
if not os.path.isdir(results_folder+'/reconstructions'):
    os.makedirs(results_folder+'/reconstructions')
if not os.path.isdir(results_folder+'/transfers'):
    os.makedirs(results_folder+'/transfers')

plt.ioff()    
evaluation_results = {}

buffers = {'signals':[], 'symbols':[], 'signal_out':[], 'signal_tf_out':[], 'symbol_out':[], 'symbol_tf_out':[]}
classification_errors = []

signal_vae.eval(); symbol_vae.eval()
for file_id, f in enumerate(testSet.files):
#    file_id = fid
    preprocessing_test = Magnitude(testSet, normalize=True, preprocessing='log')
    if testSet.metadata['pitch'][file_id] is None:
        continue  
    signals = preprocessing_test(testSet.data[file_id])
#    signals = testSet.data[file_id]
    signals = np.roll(signals, -1, axis=1)
    signals = signal_vae.format_input_data(signals)
    
    symbols = []
    for l in script_args.labels:
        symbols.append(testSet.metadata[l][file_id])
    if script_args.label_type == 'binary':
        symbols = np.concatenate(symbols, 1)
    
    with torch.no_grad():
        print('Computing file %s...'%f)
        print('- latent....')
        signal_out = signal_vae.forward(signals, sample=False)
        symbol_out = symbol_vae.forward(symbols, sample=False)
        signal_z_params = signal_out['z_params_enc'][-1][0]
        symbol_z_params = symbol_out['z_params_enc'][-1][0]
        signal_z = signal_out['z_enc']
        symbol_z = symbol_out['z_enc']
        print('- transfer....')
        signal_tf = signal_vae.decode(signal_vae.format_input_data(symbol_z), sample=False)[0]['out_params']
        symbol_tf = symbol_vae.decode(symbol_vae.format_input_data(signal_z_params), sample=False)[0]['out_params']

    buffers['signals'].append(signals);
    buffers['symbols'].append([s.clone() for s in symbols])

    buffers['signal_out'].append([s for s in signal_out['x_params']]); 
    buffers['symbol_out'].append([[s for s in sym] for sym in symbol_out['x_params']])
    buffers['signal_tf_out'].append([s for s in signal_tf]); 
    buffers['symbol_tf_out'].append([[s for s in sym] for sym in symbol_tf])
    
    # get errors
    if args.errors:
        print('- model errrors')
        signal_errors = get_signal_errors(signals, signal_out['x_params'])
        symbol_errors = get_symbolic_errors(symbols, symbol_out['x_params'], distribs = symbol_vae.pinput)
        signal_errors_tf = get_signal_errors(signals, signal_tf)
        symbol_errors_tf = get_symbolic_errors(symbols, symbol_tf, distribs = symbol_vae.pinput)
        print('\t-- signal errors : ', signal_errors)
        print('\t-- symbol errors : ', symbol_errors)
        print('\t-- signal transfer errors : ', signal_errors_tf)
        print('\t-- symbol transfer errors : ', symbol_errors_tf)
        evaluation_results[f] = [signal_errors, symbol_errors, signal_errors_tf, symbol_errors_tf]
    
    if args.classifier:
        print('- model errors')
        classifier_path = model_path+'/test_classifier.t7'
        pca = None
        try:
            loaded_data = torch.load(classifier_path, map_location=args.cuda)
            pca = loaded_data['pca']
            out_parameters = symbol_vae.pinput; hdims = symbol_vae.phidden[0]['dim']; hnum = symbol_vae.phidden[0]['nlayers']
            classifier = Classifier(pca.components_.shape[0], out_parameters, hidden_dims=hdims, hidden_layers=hnum)
            classifier.load_state_dict(loaded_data['state_dict'])
        except Exception as e:
            print('error loading classifier : %s'%e)
            pass
        
        if not pca is None:
            pca_out = pca.transform(signals)
            classifier_out = [(c,) for c in classifier(torch.from_numpy(pca_out).float())]
            classif_errors = get_symbolic_errors(symbols, classifier_out, distribs = symbol_vae.pinput)
            
            print('\t-- classifier errors : ', classif_errors)
            evaluation_results[f].append(classif_errors)
            classification_errors.append(classif_errors)
        
    # plot reconstructions
    if args.plot:
        plot_name = os.path.splitext(os.path.basename(f))[0]
        plot_reconstructions(signals, signal_out['x_params'][0], signal_tf[0], log=False, out='%s/plots/%s_reconstruction'%(results_folder, plot_name))
        plt.close('all')

#    # synthesize signals
    if args.resynthesize:
        original_x = signal_out['x_params'][0].clone()
        original_x[original_x < -1e-2] = 1e-2
        original_x = preprocessing_test.invert(original_x)
        resynthesize(original_x, out = '%s/reconstructions/%s.wav'%(results_folder, plot_name))
        transfer_x = signal_tf[0].clone()        
        transfer_x[transfer_x < -1e-2] = 1e-2
        transfer_x = preprocessing_test.invert(signal_tf[0].clone())
        resynthesize(transfer_x, out = '%s/transfers/%s.wav'%(results_folder, plot_name))
    

    
    
#%%
        
# Concatenate everything
buffers['signals'] = torch.cat(buffers['signals'], 0)
buffers['symbols'] = [ torch.cat([s[i] for s in buffers['symbols']], 0) for i in range(len(buffers['symbols'][0]))]
buffers['signal_out'] = [ torch.cat([s[i] for s in buffers['signal_out']], 0) for i in range(len(buffers['signal_out'][0]))]
new_symbol_buf = []
for j in range(len(buffers['symbol_out'][0])):
    new_symbol_buf.extend( [(torch.cat([s[j][i] for s in buffers['symbol_out']], 0),) for i in range(len(buffers['symbol_out'][j][0]))])
buffers['symbol_out'] = new_symbol_buf

buffers['signal_tf_out'] = [ torch.cat([s[i] for s in buffers['signal_tf_out']], 0) for i in range(len(buffers['signal_tf_out'][0]))]
new_symbol_buf = []
for j in range(len(buffers['symbol_tf_out'][0])):
    new_symbol_buf.extend( [(torch.cat([s[j][i] for s in buffers['symbol_tf_out']], 0),) for i in range(len(buffers['symbol_tf_out'][j][0]))])
buffers['symbol_tf_out'] = new_symbol_buf



signal_errors = get_signal_errors(buffers['signals'], buffers['signal_out'])
symbol_errors = get_symbolic_errors(buffers['symbols'], buffers['symbol_out'], distribs = symbol_vae.pinput)
signal_errors_tf = get_signal_errors(buffers['signals'], buffers['signal_tf_out'])
symbol_errors_tf = get_symbolic_errors(buffers['symbols'], buffers['symbol_tf_out'], distribs = symbol_vae.pinput)

# Classifier errors
if len(classification_errors) > 0:
    ratios= np.array([x['classif_ratio'] for x in classification_errors]) 
    classifier_ratios = np.sum(ratios, 0) / ratios.shape[0]
#%%

print('FINAL ERRORS : ')
print(" - Signal errors : ", signal_errors)
print(" - Symbol errors  : ", symbol_errors)
print(" - Transfered signal error : ", signal_errors_tf)
print(" - Transfered symbol error : ", symbol_errors_tf)
#torch.save(evaluation_results, results_folder+'/evaluation.t7')

print("Final baseline ratios : ", classifier_ratios)
