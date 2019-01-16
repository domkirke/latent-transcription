"""

 Import toolbox       : Audio dataset import

 This file contains the definition of an audio dataset import.

 Author               : Philippe Esling
                        <esling@ircam.fr>

"""
from matplotlib import pyplot as plt
    
    
import numpy as np
import scipy as sp
import os
import re
import pdb
try:
    from skimage.transform import resize
except:
    print('Skip skimage')
# Package-specific import
from . import utils as du
from . import generic
#from .signal.transforms import computeTransform
from librosa import load as lbload


"""
###################################
# Initialization functions
###################################
"""


class DatasetAudio(generic.Dataset):    
    """ Definition of a basic dataset
    Attributes:
        dataDirectory: 
    """
        
    def __init__(self, options):
        super(DatasetAudio, self).__init__(options)
        # Accepted types of files
        self.types = options.get("types") or ['mp3', 'wav', 'wave', 'aif', 'aiff', 'au'];
        self.importBatchSize = options.get("importBatchSize") or 64;
        # name of the transform imported
        self.transformName = options.get("transformName") or None;
        self.forceRecompute = options.get("forceRecompute") or False;
        # Type of audio-related augmentations
        self.augmentationCallbacks = [];
        if self.importType == "asynchronous":
            self.flattenData = self.flattenDataAsynchronous

    """
    ###################################
    # Import functions
    ###################################
    """
    
    def importData(self, idList, options, padding=False):
        """ Import part of the audio dataset (linear fashion) """
        options["transformName"] = options.get("transformName") or self.transformName;
        options["dataDirectory"] = options.get("dataDirectory") or self.dataDirectory;
        options["analysisDirectory"] = options.get("analysisDirectory") or self.analysisDirectory;
        
        # We will create batches of data
        indices = []
        
        # If no idList is given then import all !
        if (idList is None):
            indices = np.linspace(0, len(self.files) - 1, len(self.files))
            if padding:
                indices = np.pad(indices, (0, self.importBatchSize - (len(self.files) % self.importBatchSize)), 'constant', constant_values=0)
                indices = np.split(indices, len(indices) / self.importBatchSize)
            else:
                indices = [indices]
        else:
            indices = np.array(idList)
            
        # Init data
        self.data = [None] * len(self.files)
        
        # Parse through the set of batches
        for v in indices:
            curFiles = [None] * v.shape[0]
            for f in range(v.shape[0]):
                curFiles[f] = self.files[int(v[f])]
            if self.importType == "direct":
                curData, curMeta = self.importAudioData(curFiles, options)
            elif self.importType == "asynchronous":
                curData, curMeta = self.importDataAsynchronous(curFiles, options)
            for f in range(v.shape[0]):
                self.data[int(v[f])] = curData[f]

               
    def importAudioData(self, curBatch, options):
        dataPrefix = options.get('dataPrefix')
        dataDirectory = options.get('dataDirectory') or dataPrefix+'/data' or ''
        analysisDirectory = options.get('analysisDirectory') 
        if analysisDirectory is None: 
            try:
                analysisDirectory = options.get('dataPrefix')
                analysisDirectory += '/analysis'
            except TypeError:
                print('[Error] Please specify an analysis directory to import audio data')
             
                
        transformName= options.get('transformName')        
        finalData = []
        finalMeta = []
        for f in curBatch:
            if transformName == 'raw':
                finalData.append(importRawSignal(f, options))
                continue
            curAnalysisFile = re.sub(dataDirectory, analysisDirectory+'/'+transformName, f)
            curAnalysisFile = os.path.splitext(curAnalysisFile)[0] + '.npy'
                        
            finalData.append(np.load(curAnalysisFile))
            finalMeta.append(0);
        return finalData, finalMeta


    def checkInputTransforms(self):
        for i in reversed(range(len(self.files))):
            analysis_path = os.path.splitext(re.sub(self.dataDirectory, self.analysisDirectory+'/'+self.transformName, self.files[i]))[0]+'.npy'
            try:
                fiD_test = open(analysis_path)
                fiD_test.close()
            except FileNotFoundError:
                del self.files[i]
                pass
        self.hash = {self.files[i]:i for i in range(len(self.files))}
        




    ########################################
    #####  Asynchornous methods
    
    def importDataAsynchronous(self, curBatch, options):
        dataDirectory = options.get('dataDirectory')
        analysisDirectory = options.get('analysisDirectory') 
        
        if analysisDirectory is None: 
            try:
                analysisDirectory = options.get('dataPrefix')
                analysisDirectory += '/analysis'
            except TypeError:
                print('[Error] Please specify an analysis directory to import audio data')
                
        transformName= options.get('transformName')
        finalData = []
        finalMeta = []
        selector = options.get('selector_gen', asyn.selector_take)
        stride = options.get('stride')
        transpose = options.get('transpose', False)
        for f in curBatch:
            if transformName == 'raw':
                raise NotImplementedError
            #pdb.set_trace()    
            curAnalysisFile = re.sub(dataDirectory, analysisDirectory, f)
            curAnalysisFile = os.path.splitext(curAnalysisFile)[0] + '.npy'
            data = np.load(curAnalysisFile)
            if transpose:
                data = data.T
            finalData.append(asyn.OfflineDataList(curAnalysisFile, data, selector_gen = selector, stride = stride))
            finalMeta.append(0);
        return finalData, finalMeta


    def load_offline_entries(self, offline_entries):
        self.data = np.load(offline_entries)


    def save_offline_entries(self, out=None, options={}):
        out = out if out else options.get('analysisDirectory', self.analysisDirectory) 
        np.save(out+'/offline_calls.npy', self.data)

    def flattenDataAsynchronous(self, selector=lambda x: x):
        # initialize
        newData = []
        newMetadata = {}
        for k, v in self.metadata.items():
            newMetadata[k] = []
        newFiles = []
        revHash = {}
        # new hash from scratch
        newHash = dict(self.hash)
        for k, v in self.hash.items():
            newHash[k] = []
        # filter dataset
        for i in range(len(self.data)):
            # update minimum content shape
            chunk_to_add = selector(self.data[i].entries)
            newData.extend(chunk_to_add)
            for k, _ in newMetadata.items():
                newMetadata[k].extend([self.metadata[k][i]]*len(chunk_to_add))
            newFiles.extend([self.files[i]]*len(chunk_to_add))
            
        self.data = asyn.OfflineDataList(newData)
        self.metadata = newMetadata
        for k,v in newMetadata.items():
            newMetadata[k] = np.array(v)
        self.files = newFiles
        self.hash = newHash
        self.revHash = revHash


    def importRawData(self, grainSize=512, overlap = 2, ulaw=256):
        self.data = []
        for f in self.files:
            sig, sr = lbload(f)
            if ulaw > 2 :
                sig = signal.MuLawEncoding(ulaw)(sig)
            n_chunks = sig.shape[0] // (grainSize // overlap) - 1
            grainMatrix = np.zeros((n_chunks, grainSize))
            for i in range(n_chunks):
                idxs = (i*(grainSize // overlap), i*(grainSize // overlap)+grainSize)
                grainMatrix[i] = sig[idxs[0]:idxs[1]]
            self.data.append(grainMatrix)

    """
    ###################################
    # Get asynchronous pointer and options to import
    ###################################
    """
    def getAsynchronousImport(self):
        a, transformOpt = self.getTransforms()
        options = {
            "matlabCommand":self.matlabCommand,
            "transformType":self.transformType,
            "dataDirectory":self.dataDirectory,
            "analysisDirectory":self.analysisDirectory,
            "forceRecompute":self.forceRecompute,
            "transformOptions":transformOpt,
            "backend":self.backend,
            "verbose":self.verbose 
            }
        return importAudioData, options


    """
    ###################################
    # Obtaining transform set and options
    ###################################
    """
    
    def getTransforms(self):
        """
        Transforms (and corresponding options) available
        """
        # List of available transforms
        transformList = [
            'raw',                # raw waveform
            'stft',               # Short-Term Fourier Transform
            'mel',                # Log-amplitude Mel spectrogram
            'mfcc',               # Mel-Frequency Cepstral Coefficient
#            'gabor',              # Gabor features
            'chroma',             # Chromagram
            'cqt',                # Constant-Q Transform
            'gammatone',          # Gammatone spectrum
            'dct',                # Discrete Cosine Transform
#            'hartley',            # Hartley transform
#            'rasta',              # Rasta features
#            'plp',                # PLP features
#            'wavelet',            # Wavelet transform
#            'scattering',         # Scattering transform
#            'cochleogram',        # Cochleogram
            'strf',               # Spectro-Temporal Receptive Fields
            'csft',                 # Cumulative Sampling Frequency Transform
            'modulation',          # Modulation spectrum
            'nsgt',               # Non-stationary Gabor Transform
            'nsgt-cqt',               # Non-stationary Gabor Transform (CQT scale)
            'nsgt-mel',               # Non-stationary Gabor Transform (Mel scale)
            'nsgt-erb',               # Non-stationary Gabor Transform (Mel scale)
            'strf-nsgt',              # Non-stationary Gabor Transform (STRF scale)
        ];
                
        # List of options
        transformOptions = {
            "debugMode":0,
            "resampleTo":22050,
            "targetDuration":0,
            "winSize":2048,
            "hopSize":1024,
            #"nFFT":2048,
            # Normalization
            "normalizeInput":False,
            "normalizeOutput":False,
            "equalizeHistogram":False,
            "logAmplitude":False,
            #Raw features
            "grainSize":512,
            "grainHop":512,
            # Phase
            "removePhase":False,
            "concatenatePhase":False,
            # Mel-spectrogram
            "minFreq":30,
            "maxFreq":11000,
            "nbBands":128,
            # Mfcc
            "nbCoeffs":13,
            "delta":0,
            "dDelta":0,
            # Gabor features
            "omegaMax":'[pi/2, pi/2]',
            "sizeMax":'[3*nbBands, 40]',
            "nu":'[3.5, 3.5]',
            "filterDistance":'[0.3, 0.2]',
            "filterPhases":'{[0, 0], [0, pi/2], [pi/2, 0], [pi/2, pi/2]}',
            # Chroma
            "chromaWinSize":2048,
            # CQT
            "cqtBins":360,
            "cqtBinsOctave":60,
            "cqtFreqMin":64,
            "cqtFreqMax":8000,
            "cqtGamma":0.5,
            # Gammatone
            "gammatoneBins":64,
            "gammatoneMin":64,
            # Wavelet
            "waveletType":'\'gabor_1d\'',
            "waveletQ":8,
            # Scattering
            "scatteringDefault":1,
            "scatteringTypes":'{\'gabor_1d\', \'morlet_1d\', \'morlet_1d\'}',
            "scatteringQ":'[8, 2, 1]',
            "scatteringT":8192,
            # Cochleogram
            "cochleogramFrame":64,        # Frame length, typically, 8, 16 or 2^[natural #] ms.
            "cochleogramTC":16,           # Time const. (4, 16, or 64 ms), if tc == 0, the leaky integration turns to short-term avg.
            "cochleogramFac":-1,          # Nonlinear factor (typically, .1 with [0 full compression] and [-1 half-wave rectifier]
            "cochleogramShift":0,         # Shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
            "cochleogramFilter":'\'p\'',      # Filter type ('p' = Powen's IIR, 'p_o':steeper group delay)
            # STRF
            "strfFullT":0,                # Fullness of temporal margin in [0, 1].
            "strfFullX":0,                # Fullness of spectral margin in [0, 1].
            "strfBP":0,                   # Pure Band-Pass indicator
            "strfRv": np.power(2, np.linspace(0, 5, 5)),     # rv: rate vector in Hz, e.g., 2.^(1:.5:5).
            "strfSv": np.power(2, np.linspace(-2, 3, 6)),    # scale vector in cyc/oct, e.g., 2.^(-2:.5:3).
            "strfMean":0,                  # Only produce the mean activations
            "csftDensity":512,
            "csftNormalize":True
        }
        return transformList, transformOptions;
       
    def __dir__(self):
        tmpList = super(DatasetAudio, self).__dir__()
        return tmpList + ['importBatchSize', 'transformType', 'matlabCommand']
    
    
    def plotExampleSet(self, setData, labels, task, ids):
        fig = plt.figure(figsize=(12, 24))
        ratios = np.ones(len(ids))
        fig.subplots(nrows=len(ids),ncols=1,gridspec_kw={'width_ratios':[1], 'height_ratios':ratios})
        for ind1 in range(len(ids)):
            ax = plt.subplot(len(ids), 1, ind1 + 1)
            if (setData[ids[ind1]].ndim == 2):
                ax.imshow(np.flipud(setData[ids[ind1]]), interpolation='nearest', aspect='auto')
            else:
                tmpData = setData[ids[ind1]]
                for i in range(setData[ids[ind1]].ndim - 2):
                    tmpData = np.mean(tmpData, axis=0)
                ax.imshow(np.flipud(tmpData), interpolation='nearest', aspect='auto')
            plt.title('Label : ' + str(labels[task][ids[ind1]]))
            ax.set_adjustable('box-forced')
        fig.tight_layout()
        
    def plotRandomBatch(self, task="genre", nbExamples=5):
        setIDs = np.random.randint(0, len(self.data), nbExamples)
        self.plotExampleSet(self.data, self.metadata, task, setIDs)
        
   
    
           
    """
    ###################################
    # Transform functions
    ###################################
    """
    
    def computeTransforms(self, transformTypes, transformOptions, transformNames=None, idList=None, padding=False, forceRecompute=False, verbose=False):
        dataDirectory = self.dataDirectory or self.dataPrefix+'/data'
        analysisDirectory = self.analysisDirectory or self.dataPrefix+'/analysis'
        
        if not issubclass(type(transformTypes), list):
            transformTypes = [transformTypes]
        if not issubclass(type(transformOptions), list):
            transformOptions = [transformOptions]
            
        transformNames = transformNames or transformTypes;
        if not issubclass(type(transformNames), list):
            transformNames = [transformNames]
        
        if len(transformNames)!=len(transformTypes):
            raise Exception('please give the same number of transforms and names')
            
        # get indices to compute
        if (idList is None):
            indices = np.linspace(0, len(self.files) - 1, len(self.files))
            if padding:
                indices = np.pad(indices, (0, self.importBatchSize - (len(self.files) % self.importBatchSize)), 'constant', constant_values=0)
                indices = np.split(indices, len(indices) / self.importBatchSize)
            else:
                indices = [indices]
        else:
            indices = np.array(idList)
                        
        if not os.path.isdir(analysisDirectory):
            os.makedirs(analysisDirectory)
            
        for i in range(len(transformTypes)):
            current_transform_dir = analysisDirectory+'/'+transformNames[i]
            if not os.path.isdir(current_transform_dir):
                os.makedirs(current_transform_dir)
            for v in indices:
                curFiles = [None] * v.shape[0]
                for f in range(v.shape[0]):
                    curFiles[f] = self.files[int(v[f])]
                makeAnalysisFiles(curFiles, transformTypes[i], transformOptions[i], dataDirectory, analysisDirectory+'/'+transformNames[i], forceRecompute=forceRecompute, verbose=verbose)
                

            # save transform parameters
            np.save(current_transform_dir+'/transformOptions.npy', transformOptions[i])



"""
###################################
# External functions for transfroms and audio imports
###################################
"""       

from .. import nsgt3
import librosa
import scipy.signal as scs


def computeTransform(audioList, transformType, options):
    resampleTo = options.get('resampleTo') or 22050
    targetDuration = options.get('targetDuration') or 0    
    minFreq = options.get('minFreq', 30)
    maxFreq =  options.get('maxFreq', 11000)
    nsgtBins = options.get('nsgtBins', 48)
    downsampleFactor = options.get('downsampleFactor', 10)
    for i in range(len(audioList)):
        # Current audio file
        audioFile = audioList[i];
        if audioFile is None:
            continue
        
        # Read the corresponding signal
        print('     - Processing ' + audioFile)
        sig, fs = librosa.load(audioFile)
        # Turn to mono if multi-channel file
        if (len(sig.shape) > 1 and sig.shape[1] > 1):
            sig = np.mean(sig, 2)
            
        # First resample the signal (similar to ircamDescriptor)
        if (fs != resampleTo):
            sig = scs.resample(sig, int(resampleTo * (len(sig) / float(fs))))
            fs = resampleTo
            
        # Now ensure that we have the target duration
        if (targetDuration):
            # Input is longer than required duration
            if ((len(sig) / fs) > targetDuration):
                sig = sig[:int((targetDuration * fs))]
            # Otherwise pad with zeros
            else:
                sig = np.pad(sig, int(np.round(targetDuration * fs)) - len(sig), 'constant', constant_values = 0);

        scl = nsgt3.fscale.OctScale(minFreq, maxFreq, nsgtBins)
        # Calculate transform parameters
        nsgt = nsgt3.cq.NSGT(scl, resampleTo, len(sig), real=True, matrixform=True, reducedform=1)
        # forward transform 
        currentTransform = np.array(list(nsgt.forward(sig)))
        currentTransform = np.transpose(currentTransform)    

        if (downsampleFactor > 0):
            print('downsize')
            # Rescale the corresponding transform
            if (not np.iscomplexobj(currentTransform)):
                currentTransform = resize(currentTransform, (int(currentTransform.shape[0] / downsampleFactor), currentTransform.shape[1]), mode='constant')
            else:
                currentTransform_abs = np.abs(currentTransform)
                currentTransform_abs = resize(currentTransform_abs, (int(currentTransform_abs.shape[0] / downsampleFactor), currentTransform_abs.shape[1]), mode='constant')
                currentTransform_phase = np.angle(currentTransform)
                currentTransform_phase = resize(currentTransform_phase, (int(currentTransform_phase.shape[0] / downsampleFactor), currentTransform_phase.shape[1]), mode='constant')
                currentTransform = currentTransform_abs * np.exp(currentTransform_phase*1j)
    return [currentTransform]
    
    
def makeAnalysisFiles(curBatch, transformType, options, oldRoot, newRoot, transformName=None, backend='python', forceRecompute=False, verbose=False):
    transformName = transformName or transformType;
    # Initialize matlab command
    #TODO: check that
                    
    # Parse through the set of batches
    curAnalysisFiles = [None] * len(curBatch)
    audioList = [None] * len(curBatch)
    curIDf = 0
    
    # Check which files need to be computed
    for i in range(len(curBatch)):
        curFile = curBatch[i]
        analysisName = os.path.splitext(curFile.replace(du.esc(oldRoot), newRoot))[0] + '.npy'
        try:
            fIDTest = open(analysisName, 'r')
        except IOError:
            fIDTest = None
        if ((fIDTest is None) or (forceRecompute == True)):
            audioList[curIDf] = curFile 
            curAnalysisFiles[curIDf] = analysisName
            curIDf = curIDf + 1
        else: 
            fIDTest.close()
    audioList = audioList[:curIDf]
    curAnalysisFiles = curAnalysisFiles[:curIDf]
    
    # Some of the files have not been computed yet
    if (len(audioList) > 0):
        unprocessedString = ""
        if verbose:
            print("* Computing transforms ...")
        for i, target_file in enumerate(curAnalysisFiles):
            current_transforms = computeTransform([audioList[i]], 'nsgt-cqt',  options)
            if target_file is None:
                continue
            else:
                np.save(curAnalysisFiles[i], current_transforms[0])
        
    
