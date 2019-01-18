<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> <script type="text/javascript"> // Show button function look(type){ param=document.getElementById(type); if(param.style.display == "none") param.style.display = "block"; else param.style.display = "none" } </script> 

<script type="text/javascript"> 
      // Show button
      function look(type){ 
      param=document.getElementById(type); 
      if(param.style.display == "none") param.style.display = "block"; 
      else param.style.display = "none" 
      } 
</script> 

# Cross-modal variational inference for musical transcription and generation

This support page provides additional examples for the article *Cross-modal variational inference for musical transcription and generation*, submitted at [IJCNN2019](https://www.ijcnn.org/). You can find the [corresponding code here](https://github.com/domkirke/latent-transcription)

Music transcription, that consists in extracting rel- evant musical annotations from an audio signal, is still an active field of research in the domain of Musical Information Retrieval. This complex task, that is also related to other topics such as pitch extraction or instrument recognition, is a demanding subject that gave birth to numerous approaches, mostly based on advanced signal processing-based algorithms. However, these techniques are often non-generic, allowing the extraction of definite physical properties of the signal (pitch, octave), but not allowing arbitrary vocabularies or more general annotations. On top of that, these techniques are one-sided, meaning that they can extract symbolic data from an audio signal, but cannot perform the reverse process and make symbol-to-signal generation. In this paper, we propose an alternative approach to music transcription by turning this problem into a density estimation task over signal and symbolic domains, considered both as related random variables. We estimate this joint distribution with two different variational auto-encoders, one for each domain, whose inner representations are forced to match with an additive constraint. This system allows both models to learn and generate separately, but also allows signal-to-symbol and symbol-to-signal inference, thus performing musical transcription and label-constrained audio generation. In addition to its versatility, this system is rather light during training and generation while allowing several interesting creative uses.

This support page provides the following elements: 

* [Reconstructions of instrumental sound distributions](#reconstructions-of-instrumental-sound-distributions)
* Symbol-to-signal inference(#symbol-to-signal-inference)
* Signal-to-symbol inference(#signal-to-symbol-inference)
* Sound morphing and free navigation(#sound-morphing-and-free-navigation)

## Reconstructions of instrumental sound distributions

Below we show some examples of reconstructions and transfer from random excerpts from the dataset. Once the NSGT is obtained, phase is reconstructed with a Griffin-Lim algorithm with about 30 iterations ; we thus also put the inversed original NSGT to have a good comparison basis. 

<a href="javascript:look('rec_flute');" title="Flute examples">Flute examples</a>
<div id="rec_flute" style="display: none;">
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/flute/ex_1_orig.mp3" type='audio/mp3'>
</audio> (original)
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/flute/ex_1_reco.mp3" type='audio/mp3'>
</audio> (reconstructions)
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/flute/ex_1_tf_1.mp3" type='audio/mp3'>
</audio> <audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/flute/ex_1_tf_2.mp3" type='audio/mp3'>
</audio> <audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/flute/ex_1_tf_3.mp3" type='audio/mp3'>
</audio> (transfer)
</div>

<a href="javascript:look('rec_violin');" title="Violin examples">Violin examples</a>
<div id="rec_violin" style="display: none;">
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/violin/ex_2_orig.mp3" type='audio/mp3'>
</audio> (original)
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/violin/ex_2_reco.mp3" type='audio/mp3'>
</audio> (reconstructions)
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/violin/ex_2_tf_1.mp3" type='audio/mp3'>
</audio> <audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/violin/ex_2_tf_2.mp3" type='audio/mp3'>
</audio> <audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/violin/ex_2_tf_3.mp3" type='audio/mp3'>
</audio> (transfer)
</div>

<a href="javascript:look('rec_piano');" title="Piano examples">Piano examples</a>
<div id="rec_piano" style="display: none;">
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/piano/ex_2_orig.mp3" type='audio/mp3'>
</audio> (original)
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/piano/ex_2_reco.mp3" type='audio/mp3'>
</audio> (reconstructions)
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/piano/ex_2_tf_1.mp3" type='audio/mp3'>
</audio> <audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/piano/ex_2_tf_2.mp3" type='audio/mp3'>
</audio> <audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/piano/ex_2_tf_3.mp3" type='audio/mp3'>
</audio> (transfer)
</div>

<a href="javascript:look('2mx_reco');" title="2mx examples">2-mixture examples</a>
<div id="2mx_reco" style="display: none;">
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/2-mixture/ex_7_orig.mp3" type='audio/mp3'>
</audio> (original)
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/2-mixture/ex_7_reco.mp3" type='audio/mp3'>
</audio> (reconstructions)
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/2-mixture/ex_7_tf_1.mp3" type='audio/mp3'>
</audio> <audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/2-mixture/ex_7_tf_2.mp3" type='audio/mp3'>
</audio> <audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/audio_reconstructions/2-mixture/ex_7_tf_3.mp3" type='audio/mp3'>
</audio> (transfer)
</div>

## Symbol-to-signal inference
Here we show examples of symbols to signal inference : taking a combination of labels [*octave*, *pitch class*, *dynamics*], we sample the corresponding latent distirbutions and pass it through the signal decoder (here, the midi file *arpeg-FMaj-060BPM-leg-lined.mid* of the Cantos & al. dataset)

* Flute
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/signal_transfer/fl/arpeg-FMaj-060BPM-leg-lined.mp3" type='audio/mp3'>
</audio> 
* Piano
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/signal_transfer/pno/arpeg-FMaj-060BPM-leg-lined.mp3" type='audio/mp3'>
</audio> 
* Trumpet-C
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/signal_transfer/tpc/arpeg-FMaj-060BPM-leg-lined.mp3" type='audio/mp3'>
</audio> 
* Violin
<audio controls preload="auto" data-setup="{}" width="100%"> 
<source src="audio/signal_transfer/vln/arpeg-FMaj-060BPM-leg-lined.mp3" type='audio/mp3'>
</audio> 


## Signal-to-symbol inference

## Sound morphing 
Here, we take an incoming sequence of labels and transfer it into the signal domain with 1) no latent interpolation 2) linear interpolation 3) cubic interpolation. We can hear that interpolating between latent positions gives more natural transitions, and show at which point the latent space is smooth and well-organized.

<table>
  <tr>
    <td> </td>
    <td style="text-align: center; vertical-align: middle;"> No interpolation </td> 
    <td style="text-align: center; vertical-align: middle;"> Linear interpolation </td>
    <td style="text-align: center; vertical-align: middle;"> Cubic interpolation </td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;"> Alto-Sax </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/sequence/sax/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-1d/sax/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-3d/sax/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;"> Flute </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/sequence/fl/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-1d/fl/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-3d/fl/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;"> Piano </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/sequence/pno/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-1d/pno/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-3d/pno/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;"> Trumpet-C </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/sequence/tpc/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-1d/tpc/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-3d/tpc/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;"> Violin </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/sequence/vln/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-1d/vln/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls preload="auto" data-setup="{}" width="100%"> 
      <source src="audio/morph-3d/vln/sequence_3224524924112.mp3" type='audio/mp3'>
      </audio> 
    </td>
  </tr>
</table>




