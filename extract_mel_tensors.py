import argparse
import json
from env import AttrDict
import os
import librosa
import torch
from meldataset import mel_spectrogram
import re
import numpy as np

MAX_WAV_VALUE = 32768.0
path_core="/scratch/s5007453/hifi-gan-laughnet/laughter/output/"

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config_v1.json')
a = parser.parse_args()

with open(a.config) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

for item in os.listdir(path_core):
  file_path=os.path.join(path_core, item)
  
  if file_path.endswith(".wav"):
    audio, sampling_rate = librosa.load(file_path)
    audio = audio / MAX_WAV_VALUE
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    
    mel = mel_spectrogram(audio, h.n_fft, h.num_mels,
                                  h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax,
                                  center=False)
    
    fn = re.split(r"[/.]", item)[0]
    nfn = fn+".npy"
    
    np.save("/scratch/s5007453/hifi-gan-laughnet/ft_dataset/"+nfn, mel)