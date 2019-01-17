## Cross-modal variational inference for musical transcription and generation

This repository hosts the code for the article "Cross-modal variational inference for musical transcription and generation" submitted at IJCNN 2019. This code allows the training, the analysis, and various generation methods of the models presented in the article. It is divided in four main scripts :  

* `lt_train.py` allows model training
* `lt_analyze.py` performs the evaluation of the selected model 
* `lt_midi.py` performs the model evaluation over the test dataset
* `lt_play.py` allows various methods of generation using the selected model : midi import, audio import, and supervised / free navigation.


### Model training :package:
`lt_train.py` is the python script for MIDI training. It requires the training dataset, that you may find [here](http://www.wolfgang.wang/lt_set.zip). Training arguments for this script are : 

```
  -h, --help            show this help message and exit
  --dbroot DBROOT       root path of the database (given .npy file)
  --savedir SAVEDIR     output directory
  --frames [FRAMES [FRAMES ...]]
                        frames taken in each sound file (empty for all file,
                        chunk id or chunk range)
  --dims DIMS [DIMS ...]
                        number of latent dimensions
  --hidden_dims_1 HIDDEN_DIMS_1 [HIDDEN_DIMS_1 ...]
                        latent layerwise hidden dimensions for audio vae
  --hidden_num_1 HIDDEN_NUM_1 [HIDDEN_NUM_1 ...]
                        latent layerwise number of hidden layers for audio vae
  --hidden_num_2 HIDDEN_NUM_2 [HIDDEN_NUM_2 ...]
                        latent layerwise number of hidden layers for symbolic
                        vae
  --hidden_dims_2 HIDDEN_DIMS_2 [HIDDEN_DIMS_2 ...]
                        latent layerwise hidden dimensions for symbolic vae
  --labels [LABELS [LABELS ...]]
                        name of conditioning labels (octave, pitch, dynamics)
  --instruments INSTRUMENTS [INSTRUMENTS ...]
                        name of instruments
  --label_type {binary,categorical}
                        label conditioning distribution
  --regularization_type {kld,l2}
                        latent regularization type between both latent spaces
  --random_mode {constant,bernoulli}
                        random weighing of each source in the mixtures
  --zero_extra_class ZERO_EXTRA_CLASS
                        has an extra zero class when source is silent
                        (recommanded with bernoulli random_mode)
  --epochs EPOCHS       nuber of training epochs
  --save_epochs SAVE_EPOCHS
                        saving epochs
  --plot_epochs PLOT_EPOCHS
                        plotting epochs
  --name NAME           name of current training
  --cuda CUDA           cuda id (-1 for cpu)
  --load LOAD
  --load_epoch LOAD_EPOCH
  --beta_1 BETA_1       beta regularization for signal vae
  --beta_2 BETA_2       beta regularization for symbol vae
  --cross_1 CROSS_1     cross-regularization between z_signal and z_symbol
  --cross_2 CROSS_2     cross-regularization between z_symbol and z_signal
  --adversarial ADVERSARIAL
                        set to 1 for adversarial reinfocement.
  --adv_dim ADV_DIM     hidden capacity for adversarial network
  --adv_num ADV_NUM     number of hidden networks for adversarial network
  --adv_lr ADV_LR       learning rate for adversarial network
  ```



### Model analysis


### Model MIDI import 


### Model play

