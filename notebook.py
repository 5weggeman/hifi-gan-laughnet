#!/usr/bin/env python
# coding: utf-8

# ### Import relevant modules
# `WaveformSilhouette` and `LinearEncoding` are defined in `silhouette.py`

import torch
import torchaudio
import torch.nn.functional as F
import random

#import matplotlib.pyplot as plt

from silhouette import WaveformSilhouette, LinearEncoding

# ### Load waveform data
# In this example we only load a single waveform with shape `[C x T]`, but usually in training we load a batch with shape `[B x C x T]`

def quant_ws(waveforms, win_length, hop_length):

  scaling_factor = random.uniform(0.3, 1.0)
  waveforms = waveforms * scaling_factor
  transform = WaveformSilhouette(win_length, hop_length)
  silh = transform(waveforms)

  #print(f"silh Shape: {silh.shape} | silh: {silh}")
  #plt.plot(silh[0,:])
  #plt.plot(silh[1,:])
  #plt.ylim([-1.0,1.0])
  #plt.show()

  # ### Quantization
  # Waveform Silhouette is a two-dimensional continuous values we want to quantize them into discrete value using either linear or mulaw encoding.
  
  quantizers = [
  #    LinearEncoding(quantization_channels=256),
  #    torchaudio.transforms.MuLawEncoding(quantization_channels=256),
      torchaudio.transforms.MuLawEncoding(quantization_channels=16)
  ]

  for quantizer in quantizers:
    silh_quan = quantizer(silh)
    #plt.plot(silh_quan[0,:])
    #plt.plot(silh_quan[1,:])
    #plt.show()


  # ### Embedding
  # We convert discrete (`Long`) value into One-hot Vector and use it as feature input.

  feat = silh_quan.long()
  
  B, X = silh_quan.shape
  C = 2
  T = int(X/2)
  #print(f"B: {B} | C: {C} | T: {T}")
  
  feat = feat.reshape((B, 2, T))
  feat = feat.permute((0, 2, 1))
  feat = F.one_hot(feat, num_classes=16)
  
  QC = feat.size(dim=-1)
  #print(f"QC: {QC}")
  C=C*QC
  
  feat = feat.reshape((B, T, C))
  feat = feat.permute((0, 2, 1))
  #print(f"feat Shape: {feat.shape}")
  #print(f"Frame Example: {feat[:,1]}")
  
  return feat

# ### End.