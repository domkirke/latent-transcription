#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:11:17 2018

@author: chemla
"""
import os, librosa
import numpy as np
from ..nsgt3 import NSGT
from ..nsgt3.fscale import OctScale, MelScale, LogScale

def regenerateAudioNSGT(data, minFreq = 30, maxFreq = 11000, nsgtBins = 48, sr = 22050, scale = 'oct', targetLen = int(3 * 22050), iterations = 100, momentum=False, testSize=False, curName = None, **kwargs):
    # Create a scale
    if (scale == 'oct'):
        scl = OctScale(minFreq, maxFreq, nsgtBins)
    if (scale == 'mel'):
        scl = MelScale(minFreq, maxFreq, nsgtBins)
    if (scale == 'log'):
        scl = LogScale(minFreq, maxFreq, nsgtBins)
    # Create the NSGT object
    nsgt = NSGT(scl, sr, targetLen, real=True, matrixform=True, reducedform=1)
    # Run a forward test
    if (testSize):
        testForward = np.array(list(nsgt.forward(np.zeros((targetLen)))))
        targetFrames = testForward.shape[1]
        nbFreqs = testForward.shape[0]
        print(data.shape, nbFreqs, targetFrames)
        assert(data.shape[0] == nbFreqs)
        assert(data.shape[1] == targetFrames)
    # Now Griffin-Lim dat
    print('Start Griffin-Lim')
    p = 2 * np.pi * np.random.random_sample(data.shape) - np.pi
    for i in range(iterations):
        S = data * np.exp(1j*p)
        inv_p = np.array(list(nsgt.backward(S)))#transformHandler(S, transformType, 'inverse', options)
        new_p = np.array(list(nsgt.forward(inv_p)))#transformHandler(inv_p, transformType, 'forward', options)
        new_p = np.angle(new_p)
        # Momentum-modified Griffin-Lim
        if (momentum):
            p = new_p + ((i > 0) * (0.99 * (new_p - p)))
        else:
            p = new_p
        # Save the output
    if not curName is None:
        librosa.output.write_wav(curName + '.wav', inv_p, sr)
    return inv_p


def audio_reconstructions(dataset, model, transform='nsgt-cqt', mode='full_files', n_files=1, transformOptions = None, preprocessing=None, partition=None, out='', plot=True, reduction=None, **kwargs):
    max_len = 3.0    
    reconstruction_methods = {'nsgt-cqt':regenerateAudioNSGT}
    rec_method = reconstruction_methods[transform]
    
    #### import data
    if mode == 'full_files':
        file_list = np.array(list(set(dataset.files)))
        current_files = file_list[np.random.permutation(file_list.shape[0])[:n_files]]
        current_names = [os.path.splitext(os.path.basename(f))[0] for f in current_files]
        current_data = []; current_lengths = []
        for i,f in enumerate(current_files):
            print('translating %s ...'%current_names[i])
            current_array, sr = librosa.load(f)
            if current_array.ndim > 1:
                current_array = np.sum(current_array, 0) / 2
            current_array = current_array[:min(current_array.shape[0], int(max_len*sr))]
            current_lengths.append(int(current_array.shape[0]))
            current_transform = transforms.transformHandler(current_array, transform, 'forward', transformOptions)
            current_data.append(current_transform)
            
    #### forward data
    current_output = []; zs = []
    for i, d in enumerate(current_data):
        if not preprocessing is None:
            d = preprocessing(d)
        vae_out = model.forward(model.format_input_data(d))
        zs.append(vae_out['z_params_enc'][-1][0].detach().numpy())
        current_output.append(vae_out['x_params'][0].detach().numpy())
            
    for i, data in enumerate(current_data):
        resyn = current_output[i]
        if not preprocessing is None:
            resyn  = preprocessing.invert(resyn)
        print('inverting original transform %s...'%current_names[i])
        rec_method(np.abs(data).T, curName=out+'/'+current_names[i]+'_direct', sr=sr, testSize=True, targetLen = current_lengths[i], **transformOptions)
        print('inverting reconstructed transform %s...'%current_names[i])
        rec_method(resyn.T, curName=out+'/'+current_names[i]+'_resyn', sr=sr, testSize=True, targetLen = current_lengths[i], **transformOptions)
        if plot:
            fig, axes = lplt.plot_latent_path(zs[i], np.abs(data), resyn, reduction = reduction)
            fig.savefig(out+'/'+current_names[i]+'.pdf', format='pdf')
                
            
            
            
    
            
            