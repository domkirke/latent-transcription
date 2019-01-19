
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:18:19 2018

@author: chemla
"""
import matplotlib
matplotlib.use('agg')
import copy, argparse, numpy as np, torch, os, torch.nn as nn, sys

from skimage.transform import resize
import librosa
from sklearn.metrics import confusion_matrix

import lt
from lt.data.audio import DatasetAudio
import lt.data.metadata as mc

from lt.criterions.criterion_logdensities import log_normal, log_bernoulli, log_categorical
from lt.utils.onehot import fromOneHot, oneHot
from lt.monitor.visualize_dimred import PCA
from lt.modules.modules_classifier import Classifier
import lt.monitor.synthesize_audio as audio
from lt.criterions.criterion_scan import SCANLoss
from lt.utils.dataloader import MixtureLoader

parser = argparse.ArgumentParser()
# import arguments
parser.add_argument('-d', '--dbroot', type=str, help='dataset path', default='lt_set.npz')
parser.add_argument('-m', '--models', type=str, nargs='+', help='model to load')
parser.add_argument('-c', '--cuda', type=int, help='cuda device (leave -1 for GPU)', default=-1)
parser.add_argument('-o', '--output', type=str, help='results output')
parser.add_argument('-n', '--nb_passes', type=int, help='number of passes for loss computation (useful in case of random mixtures', default=1)
# classifier arguments
parser.add_argument('--classifier_epochs', type=int, help='number of passes for loss computation', default=1000)
parser.add_argument('--classifier_bs', type=int, help='batch size for classifier trainingloss computation', default=64)
# analysis arguments
parser.add_argument('--make_losses', type=int, default=1 , help='compute losses')
parser.add_argument('--make_figures', type=int, default=1 , help='make figures')
parser.add_argument('--make_classifier', type=int, default=1, help='train baseline classifier')
parser.add_argument('--evaluate_classifier', type=int, default=1, help='evaluate baseline classifier')
parser.add_argument('--generate_audio', type=int, default=4, help='generate audio_examples (nb of examples)')
parser.add_argument('--n_samples', type=int, default = 4, help="number of signal to symbol samples")

args = parser.parse_args()

if args.cuda < 0:
    args.cuda = 'cpu'
    

#%% Utils functions

def get_subdatasets(audioSet, args, partitions=None, preprocessing=None):
    datasets = []; 
    instrument_ids = [audioSet.classes['instrument'][d] for d in args.instruments]
    for n, iid in enumerate(instrument_ids):
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


#%% Distance functions
    
def log_prob_signal(signal, reconstruction):
    return log_normal(signal, reconstruction)

def isd(signal, reco):
    return (1/(2*np.pi))*(torch.exp(signal)/torch.exp(reco) - signal + reco - 1).sum()/(signal.shape[0])
     
def log_prob_classif(labels, reconstruction, distrib=None):
    if issubclass(type(labels), list):
        if distrib[0] == torch.distributions.Categorical:
            losses = [log_prob_classif(labels[i], reconstruction[i], distrib[i]) for i in range(len(labels))]
            averaged_losses = [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)]
            n_inst = len(losses)//3;
            for i in range(n_inst):
                averaged_losses[0] += losses[i*3]
                averaged_losses[1] += losses[i*3+1]
                averaged_losses[2] += losses[i*3+2]      
            averaged_losses = [l/n_inst for l in averaged_losses]
            return averaged_losses
        else:
            return [log_prob_classif(labels[i], reconstruction[i], distrib[i]) for i in range(len(labels))]
    else:
        if distrib == torch.distributions.Categorical:
            return log_categorical(labels, reconstruction)
        elif distrib == torch.distributions.Bernoulli:
            return log_bernoulli(labels, reconstruction)
        else:
            return 
    
def get_confusion_matrix(symbols, reco):
    if issubclass(type(symbols), list):
        cf = [get_confusion_matrix(symbols[i], reco[i]) for i in range(len(symbols))]
    else:
        if issubclass(type(reco), tuple):
            reco = reco[0]
        cf = confusion_matrix(fromOneHot(symbols), fromOneHot(reco))
    return cf

def cf2csr(cm, distrib=None):
    if issubclass(type(cm), list):
        cm = [cf2csr(cm[i]) for i in range(len(cm))]
        if distrib[0] == torch.distributions.Categorical:
            averaged_losses = [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)]
            n_inst = len(cm)//3;
            for i in range(n_inst):
                averaged_losses[0] += cm[i*3]*(1/n_inst)
                averaged_losses[1] += cm[i*3+1]*(1/n_inst)
                averaged_losses[2] += cm[i*3+2]*(1/n_inst)   
            return averaged_losses
    else:
        cm = np.sum(np.diagonal(cm))/(np.sum(cm))
    return cm

def get_signal_errors(signal, reco):
    errors = dict()
    errors['spectral-ll'] = log_prob_signal(signal, reco)
    errors['isd'] = isd(signal, reco[0])
    return errors
    
def csr_perminv(symbols, reco, n_labels):
    symbols_label = [ fromOneHot(s).detach().numpy() for s in symbols ]
    symbols_reco_label = [ fromOneHot(s[0]).detach().numpy() for s in reco ]
    
    scores = []
    full_labels = []
    for i in range(n_labels):
        full_labels.append(np.array([ symbols_label[j*n_labels + i] for j in range(len(symbols) // n_labels)]))
    for i in range(len(symbols_reco_label)):
        matches = np.zeros((symbols_reco_label[i].shape[0]))
        for j in range(symbols_reco_label[i].shape[0]):
            matches[j] = symbols_reco_label[i][j] in set(full_labels[i%n_labels][:, j].tolist())
        scores.append(matches)
    ratios = [ np.where(s == 1.)[0].shape[0] / s.shape[0] for s in scores ]
    ratios = [sum(ratios[i::3])/(len(ratios)//3) for i in range(3)]
    return ratios

def get_symbolic_errors(symbols, reco,  distribs=None, n_labels=3):
    errors = dict()
    errors['symbolic-ll'] = log_prob_classif(symbols, reco, distribs)
    errors['confusion'] = get_confusion_matrix(symbols, reco)
    errors['classif_ratio'] = cf2csr(errors['confusion'], distribs) 
    errors['classif_ratio_invariant'] = csr_perminv(symbols, reco, n_labels=n_labels)
    return errors
    
def average_errors(error_dict):
    averaged_errors = {}
    nb_passes = len(error_dict)
    for i, current_dict in enumerate(error_dict):
        for k in current_dict.keys():
            if averaged_errors.get(k):
                if issubclass(type(averaged_errors[k]), list):
                    averaged_errors[k] = [averaged_errors[k][i] + current_dict[k][i] / nb_passes for i in range(len(averaged_errors[k]))]
                else:
                    averaged_errors[k] += current_dict[k] / nb_passes
            else:
                if issubclass(type(current_dict[k]), list):
                    averaged_errors[k] = [err / nb_passes for err in current_dict[k]]
                else:
                    averaged_errors[k] = current_dict[k] / nb_passes
    return averaged_errors




    
#%% Reconstruction functions
        
def resynthesize(x_in, grain_len = 2.0, upsampleFactor = 10, out=None, preprocessing=None, threshold = 1e-4, normalize=True):
    NSGT_LENGTHS = {0.5:157, 1.0:313, 2.0:626}
    target_len = int(grain_len * 22050)
    nsgt_chunk_size = int(np.ceil(grain_len*313))
    
    # upsample incoming transform
    x_in = x_in.cpu().detach().numpy()
    current_max = np.amax(np.abs(x_in))
    x_in = resize(x_in/current_max, (int(x_in.shape[0]*upsampleFactor), x_in.shape[1]))
    x_in = x_in * current_max
    
    # invert preprocessing
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


#transformOptions = np.load(audioSet.analysisDirectory+'/nsgt-cqt/transformOptions.npy')[None][0]
#%%
        
if __name__ == '__main__':
    
    # package translation hack
    sys.modules['models']=lt
    sys.modules['models.vaes']=lt.modules
    sys.modules['models.vaes.vae_vanillaVAE'] = lt.modules.modules_vanillaVAE
    sys.modules['data'] = lt.data

    audioSet = DatasetAudio.load(args.dbroot)

    for model in args.models:
        # Load models
        print('-- MODEL : ', model)
        filename = os.path.basename(model).split('__')[0]+'_best'
        print('load file : %s'%filename)
        device = torch.device(args.cuda)
        device_name = "cpu" if device.type == 'cpu' else 'cuda:%d'%device.index
        # Get model and model properties
        basename = os.path.splitext(os.path.basename(model))[0]
        signal_data = torch.load(model+'/vae_1/%s.t7'%(filename), map_location=device_name)
        signal_vae = signal_data['class'].load(signal_data)
        symbol_data = torch.load(model+'/vae_2/%s.t7'%(filename), map_location=device_name)
        symbol_vae = signal_data['class'].load(symbol_data)
        loss = signal_data['loss']; epoch = signal_data['epoch']; partitions = signal_data['partitions']
        preprocessing = signal_data['preprocessing']; training_args = signal_data['script_args']
        random_mode = training_args.random_mode if  hasattr(training_args, 'random_mode') else 'constant'
        random_mode = 'constant'
        
        if issubclass(type(symbol_vae.pinput), list):
            symbol_dists = [p['dist'] for p in symbol_vae.pinput]
        else:
            symbol_dists = symbol_vae.pinput['dist'] 
            
        print('signal model structure : %s\n%s\n%s'%(signal_vae.pinput, signal_vae.phidden, signal_vae.platent))
        print('symbol model structure : %s\n%s\n%s'%(symbol_vae.pinput, symbol_vae.phidden, symbol_vae.platent))
 
        print('--- current epoch : ', signal_data['epoch'])

        # Import dataset
        current_datasets = get_subdatasets(audioSet, training_args, preprocessing=None, partitions=partitions)
        meta_datasets = get_symbolic_datasets(audioSet, current_datasets, training_args)
        signal_vae.eval(); symbol_vae.eval();
        
        # Initialize evaluation
        rec_errors = []; symbol_errors = []; rec_errors_tf = []; symbol_errors_tf = []; 
        mixture_buff = []; symbols_buff = [];
        nb_passes = args.nb_passes if len(current_datasets) > 1 else 1
        
        for n in range(nb_passes):
            loader = MixtureLoader(current_datasets, batch_size=None, partition='train', random_mode = random_mode)
            for mixture, x, y in loader:
                # Load training data
                mixture = preprocessing(mixture)
                mixture = signal_vae.format_input_data(mixture)
                mixture_buff.append(mixture)
                
                # symbol input
                symbols = []
                for d in range(len(meta_datasets)):
                    if issubclass(type(meta_datasets[d].data), list):
                        current_metadata = [x_tmp[loader.current_ids[d]] for x_tmp in meta_datasets[d].data] 
                        if signal_data['script_args'].zero_extra_class:
                            for i_tmp, tmp in enumerate(current_metadata):
#                                current_metadata[i_tmp] = np.concatenate((tmp, np.zeros((tmp.shape[0], 1))), 1)
                                current_metadata[i_tmp][np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata[i_tmp].shape[1]-1)+[1])
                        symbols.extend(current_metadata)
                    else:
                        current_metadata = meta_datasets[d].data[loader.current_ids[d]]
                        if signal_data['script_args'].zero_extra_class:
#                            current_metadata = np.concatenate((current_metadata, np.zeros((current_metadata.shape[0], 1))), 1)
                            current_metadata[np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata[i_tmp].shape[1]-1)+[1])
                        symbols.append(current_metadata)
                symbols = symbol_vae.format_input_data(symbols)
                
                if len(symbols_buff) == 0:
                    symbols_buff = symbols
                else:
                    for i in range(len(symbols)):
                        symbols_buff[i] = torch.cat((symbols_buff[i], symbols[i]), dim=0)
                    
                # Forward!!
                if args.make_losses:
                    with torch.no_grad():
                        # Compute forward passes for both vaes  
                        out_signal = signal_vae.forward(mixture, sample=False)
                        out_symbol = symbol_vae.forward(symbols, sample=False)
                        # Compute transfers
                        z_signal = out_signal['z_enc'][0]; z_symbol = out_symbol['z_enc'][0]
                        out_tf_signal = signal_vae.decode(signal_vae.format_input_data(z_symbol), sample=False)
                        out_tf_symbol = symbol_vae.decode(symbol_vae.format_input_data(z_signal), sample=False)
                        
                        # Compute errors
                        rec_errors.append(get_signal_errors(mixture, out_signal['x_params']))
                        symbol_errors.append(get_symbolic_errors(symbols, out_symbol['x_params'], distribs = symbol_dists))
                        rec_errors_tf.append(get_signal_errors(mixture, out_tf_signal[0]['out_params']))
                        symbol_errors_tf.append(get_symbolic_errors(symbols, out_tf_symbol[0]['out_params'], distribs = symbol_dists))
     
        
        
        # average and record losses
        if args.make_losses:
            rec_errors = average_errors(rec_errors)
            symbol_errors = average_errors(symbol_errors)
            rec_errors_tf = average_errors(rec_errors_tf)
            symbol_errors_tf = average_errors(symbol_errors_tf)
            torch.save({'rec_errors':rec_errors, 'symbol_errors':symbol_errors, 
                        'rec_errors_tf':rec_errors_tf, 'symbol_errors_tf':symbol_errors_tf}, model+'/final_errors.t7')
            print('Signal reconstruction errors :')
            print(rec_errors)
            print('Symbolical reconstruction errors :')
            print(symbol_errors)
            print('Signal transfer errors :')
            print(rec_errors_tf)
            print('Symbolical transfer errors :')
            print(symbol_errors_tf)
           
            
            
            
        # Generate audio examples
        if args.generate_audio:
            
            def plot_reconstruction(mixture, reco, tfs):
                fig = plt.figure()
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.plot(mixture.numpy())
                ax1.plot(reco.numpy(), linewidth=0.8)
                ax2 = fig.add_subplot(2, 1, 2)
                ax2.plot(mixture.numpy())
                for i in range(len(tfs)):
                    ax2.plot(tfs[i].numpy(), linewidth=0.5, c='r')
                return fig
            
            output_folder = model+'/audio_examples'
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            
            loader = MixtureLoader(current_datasets, batch_size=args.generate_audio, random_mode = random_mode)
            mixture, x, y  = next(loader.__iter__())
            mixture = preprocessing(mixture)
            mixture = signal_vae.format_input_data(mixture)
            mixture_buff.append(mixture)
            symbols = []
            for i, sd in enumerate(meta_datasets):
                if issubclass(type(sd.data), list):
                    symbols.extend([s[loader.current_ids[i]] for s in sd.data])
                else:
                    symbols.append(sd.data[loader.current_ids[i]])
            symbols = symbol_vae.format_input_data(symbols)
            if len(symbols_buff) == 0:
                symbols_buff = symbols
            else:
                for i in range(len(symbols)):
                    symbols_buff[i] = torch.cat((symbols_buff[i], symbols[i]), dim=0)
                    
            with torch.no_grad():
                out_signal = signal_vae.forward(mixture, sample=False)
                out_symbol = symbol_vae.forward(symbols, sample=False)
                # Compute transfers
                z_symbol = out_symbol['z_params_enc'][0]
                distrib = symbol_vae.platent[-1]['dist']
                out_tf_signal = []
                for s in range(args.n_samples):
                    distrib = symbol_vae.platent[-1]['dist']
                    z_signal = distrib(*tuple(z_symbol)).sample()
                    out_tf_signal.append(signal_vae.decode(signal_vae.format_input_data(z_signal), sample=False))

                # Resynthesize audio files
                for i in range(mixture.shape[0]):
                    resynthesize(mixture[np.newaxis, i], grain_len = 0.5, upsampleFactor = 500, out=output_folder+'/ex_%d_orig.wav'%i, preprocessing=preprocessing)
                    resynthesize(out_signal['x_params'][0][np.newaxis, i], grain_len = 0.5, upsampleFactor = 500, out=output_folder+'/ex_%d_reco.wav'%i, preprocessing=preprocessing)
                    for s in range(args.n_samples):
                        resynthesize(out_tf_signal[s][0]['out_params'][0][np.newaxis, i], grain_len = 1.0, upsampleFactor = 500, out=output_folder+'/ex_%d_tf_%d.wav'%(i,s), preprocessing=preprocessing)
                    fig = plot_reconstruction(mixture[i], out_signal['x_params'][0][i], [o[0]['out_params'][0][i] for o in out_tf_signal])
                    fig.savefig(output_folder+'/ex_%d.pdf'%i)




        # Train classifier
        classifier = None
        full_mixtures = torch.cat(mixture_buff, dim=0)     
        train_ids = None; test_ids = None;
        if args.make_classifier:
            # perform embedding
            pca_dim = signal_vae.platent[-1]['dim']
            pca = PCA(n_components=pca_dim)
            full_mixtures_emb = pca.fit_transform(full_mixtures)
            # make classifier
            out_parameters = symbol_vae.pinput; hdims = symbol_vae.phidden[0]['dim']; hnum = symbol_vae.phidden[0]['nlayers']
            classifier = Classifier(pca_dim, out_parameters, hidden_dims=hdims, hidden_layers=hnum)
            
 
            if args.cuda != 'cpu' and torch.cuda.is_available():
                classifier.cuda(args.cuda)
            classifier.train()
            full_mixtures_emb = torch.tensor(full_mixtures_emb).float()       
            
            # train classifier
            nb_epochs = args.classifier_epochs; batch_size = args.classifier_bs
            optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)        
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') 
            for epoch in range(nb_epochs):
                epoch_loss = torch.tensor(0., device=device, requires_grad=False)
                loader = MixtureLoader(current_datasets, batch_size=64, partition='train', random_mode = random_mode)
                #    reconstructions = []; transfers = []; 
                for mixture, x, y in loader:
                    mixture = preprocessing(mixture)
                    mixture = signal_vae.format_input_data(mixture)
                    mixture_buff.append(mixture)
                    
                    symbols = []
                    for i, sd in enumerate(meta_datasets):
                        if issubclass(type(sd.data), list):
                            symbols.extend([s[loader.current_ids[i]] for s in sd.data])
                        else:
                            symbols.append(sd.data[loader.current_ids[i]])
                    symbols = symbol_vae.format_input_data(symbols)
    
                    mixture = torch.from_numpy(pca(mixture)).float()#.cuda(args.cuda)
                    out = classifier(mixture)
                    losses = torch.zeros(len(out_parameters), device=device)
                    for i, out_dim in enumerate(out_parameters):
    #                    pdb.set_trace()
                        if out_dim['dist'] == torch.distributions.Categorical:
                            losses[i] = torch.nn.functional.cross_entropy(out[i], fromOneHot(symbols[i]))
                        elif out_dim['dist'] == torch.distributions.Bernoulli:
                            losses[i] = torch.nn.functional.binary_cross_entropy(out[i], symbols_buff[i][ids])
                            
                    batch_loss = losses.sum()
                    batch_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss = epoch_loss + batch_loss
    
                scheduler.step(epoch_loss) 
                if epoch % 20 == 0:
                    print('epoch %d - losses : %s'%(epoch, losses))
                if epoch % 100 == 0:
                    torch.save({'state_dict':classifier.state_dict(), 'pca':pca}, model+'/test_classifier.t7')
    
    
    
        # evaluate classifier
        if args.evaluate_classifier:
            if not classifier:
                classifier_dict = torch.load(model+'/test_classifier.t7', map_location='cpu')
                out_parameters = symbol_vae.pinput; hdims = symbol_vae.phidden[0]['dim']; hnums = symbol_vae.phidden[0]['nlayers']
                latent_dim = symbol_vae.platent[-1]['dim']
                classifier = Classifier(latent_dim, out_parameters, hidden_dims=hdims, hidden_layers=hnums)
                classifier.load_state_dict(classifier_dict['state_dict'])
                if args.cuda != 'cpu':
                    classifier.cuda(args.cuda)
                pca = classifier_dict['pca']
    
            loader = MixtureLoader(current_datasets, batch_size=None, partition='test', random_mode = random_mode)
            for mixture, x, y in loader:
                full_mixtures_emb = pca.transform(mixture)
                full_mixtures_emb = torch.tensor(full_mixtures_emb).float()      
                symbols = []
                for i, sd in enumerate(meta_datasets):
                    if issubclass(type(sd.data), list):
                        symbols.extend([s[loader.current_ids[i]] for s in sd.data])
                    else:
                        symbols.append(sd.data[loader.current_ids[i]])
    
                with torch.no_grad():
                    classification_out = classifier(full_mixtures_emb)
            symbols = [torch.from_numpy(c) for c in symbols]
    
#            cf_matrices = get_confusion_matrix(symbols, classification_out)
#            classif_error = cf2csr(cf_matrices, distrib=symbol_vae.pinput)
#            torch.save(classif_error, model+'/baseline_errors.t7')
#            print(classif_error)``
            classification_out = [(c,) for c in classification_out]
            errors = get_symbolic_errors(symbols, classification_out, symbol_dists)

            print('BASELINE ERRORS : ')
            print(errors)
   