#
# Copyright (c) 2022 
#   Hieu-Thi Luong, National Institute of Informatics
#   MIT Licence (https://github.com/nii-yamagishilab/waveform-silhouette-module/blob/main/LICENSE)
#

#!/usr/bin/env python
# coding: utf-8

# Import relevant modules
# `WaveformSilhouette` and `LinearEncoding` are defined in `silhouette.py`

import torch
import torchaudio
import torch.nn.functional as F
import random

from silhouette import WaveformSilhouette, LinearEncoding

def quant_ws(waveforms, win_length, hop_length):
  scaling_factor = random.uniform(0.3, 1.0)
  waveforms = waveforms * scaling_factor
  transform = WaveformSilhouette(win_length, hop_length)
  silh = transform(waveforms)
  
  # Quantization
  # Waveform Silhouette is a two-dimensional continuous values we want to quantize them into discrete value using either linear or mulaw encoding.
  # Weggeman: I commented out the first two, because I decided to use only 4-bit mu-law quantization

  quantizers = [
    #LinearEncoding(quantization_channels=256),
    #torchaudio.transforms.MuLawEncoding(quantization_channels=256),
    torchaudio.transforms.MuLawEncoding(quantization_channels = 16)
  ]

  for quantizer in quantizers:
    silh_quan = quantizer(silh)
  
  # Embedding
  # We convert discrete (`Long`) value into One-hot Vector and use it as feature input.
  
  C, T = silh_quan.shape

  feat = torch.flatten(torch.transpose(silh_quan, 0, 1))
  feat = F.one_hot(feat, num_classes = 16)
  feat = torch.transpose(feat.reshape(T, C * 16), 0, 1)

  return feat