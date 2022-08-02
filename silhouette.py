# 
# Copyright (c) 2022 
#   Hieu-Thi Luong, National Institute of Informatics
#   MIT Licence (https://github.com/nii-yamagishilab/waveform-silhouette-module/blob/main/LICENSE)
#

import torch
import torch.nn.functional as F


class WaveformSilhouette(torch.nn.Module):
  def __init__(
    self,
    win_length,
    hop_length
  ):
    super(WaveformSilhouette, self).__init__()
    self.maxpool = torch.nn.MaxPool1d(win_length, stride=hop_length, padding=(win_length - hop_length)//2)

  def forward(self, waveform):
    featmax = self.maxpool(waveform)
    featmin = -self.maxpool(-waveform)
    feat = torch.cat((featmax,featmin), axis=1)
    return feat


class LinearEncoding(torch.nn.Module):
  def __init__(
    self,
    quantization_channels=256
  ):
    super(LinearEncoding, self).__init__()
    self.quantization_channels = quantization_channels

  def forward(self, waveform):
    safe_waveform = torch.minimum(torch.abs(waveform), waveform*0.0 + 1.0)
    signal = (torch.sign(waveform)*safe_waveform+1) / 2
    signal = signal * (self.quantization_channels-1) + 0.5
    return signal.long()