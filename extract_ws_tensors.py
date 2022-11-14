import argparse
import json
from env import AttrDict
import os
import librosa
import torch
import re
import numpy as np
from waveform_silhouette import quant_ws

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
    
    ws = quant_ws(audio, h.win_size, h.hop_size).float()
    
    fn = re.split(r"[/.]", item)[0]
    nfn = fn+".npy"
    
    np.save("/scratch/s5007453/hifi-gan-laughnet/ft_dataset/"+nfn, ws)